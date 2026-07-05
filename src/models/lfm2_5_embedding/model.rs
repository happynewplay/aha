use std::collections::HashMap;

use anyhow::{Result, anyhow};
use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::{
    Conv1d, Embedding, Linear, Module, RmsNorm, VarBuilder, embedding, linear_no_bias, rms_norm,
};

use crate::{
    models::{
        EmbeddingOptions, EmbeddingPromptName,
        common::{conv1d_depthwise, eager_attention_forward, get_conv1d},
        lfm2_5::{
            config::Lfm2_5Config,
            model::{Lfm2_5LayerKind, validate_layer_types},
        },
        lfm2_5_embedding::config::{Lfm2_5EmbeddingConfig, Lfm2_5EmbeddingPoolingStrategy},
    },
    position_embed::rope::{RoPE, apply_rotary_pos_emb},
    tokenizer::TokenizerModel,
    utils::{find_type_files, get_device, get_dtype},
};

fn l2_normalize(values: &mut [f32]) {
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in values.iter_mut() {
            *value /= norm;
        }
    }
}

struct Lfm2_5EmbeddingMLP {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl Lfm2_5EmbeddingMLP {
    fn new_from_vb(vb: VarBuilder, cfg: &Lfm2_5Config) -> Result<Self> {
        let intermediate_size = cfg.adjusted_intermediate_size();
        let w1 = linear_no_bias(cfg.hidden_size, intermediate_size, vb.pp("w1"))?;
        let w2 = linear_no_bias(intermediate_size, cfg.hidden_size, vb.pp("w2"))?;
        let w3 = linear_no_bias(cfg.hidden_size, intermediate_size, vb.pp("w3"))?;
        Ok(Self { w1, w2, w3 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.w1.forward(xs)?.silu()?;
        let up = self.w3.forward(xs)?;
        let hidden = gate.broadcast_mul(&up)?;
        Ok(self.w2.forward(&hidden)?)
    }
}

struct Lfm2_5EmbeddingAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    q_layernorm: RmsNorm,
    k_layernorm: RmsNorm,
    num_heads: usize,
    num_key_value_heads: usize,
    num_key_value_groups: usize,
    head_dim: usize,
}

impl Lfm2_5EmbeddingAttention {
    fn new_from_vb(vb: VarBuilder, cfg: &Lfm2_5Config) -> Result<Self> {
        if cfg.hidden_size % cfg.num_attention_heads != 0 {
            return Err(anyhow!(
                "hidden_size {} is not divisible by num_attention_heads {}",
                cfg.hidden_size,
                cfg.num_attention_heads
            ));
        }
        if cfg.num_attention_heads % cfg.num_key_value_heads != 0 {
            return Err(anyhow!(
                "num_attention_heads {} is not divisible by num_key_value_heads {}",
                cfg.num_attention_heads,
                cfg.num_key_value_heads
            ));
        }

        let head_dim = cfg.head_dim();
        let q_proj = linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(
            cfg.hidden_size,
            cfg.num_key_value_heads * head_dim,
            vb.pp("k_proj"),
        )?;
        let v_proj = linear_no_bias(
            cfg.hidden_size,
            cfg.num_key_value_heads * head_dim,
            vb.pp("v_proj"),
        )?;
        let out_proj = linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp("out_proj"))?;
        let q_layernorm = rms_norm(head_dim, cfg.norm_eps, vb.pp("q_layernorm"))?;
        let k_layernorm = rms_norm(head_dim, cfg.norm_eps, vb.pp("k_layernorm"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            q_layernorm,
            k_layernorm,
            num_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            num_key_value_groups: cfg.num_attention_heads / cfg.num_key_value_heads,
            head_dim,
        })
    }

    fn forward(&self, xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (b_size, seq_len, _) = xs.dims3()?;
        let query_states =
            self.q_proj
                .forward(xs)?
                .reshape((b_size, seq_len, self.num_heads, self.head_dim))?;
        let query_states = self.q_layernorm.forward(&query_states)?.transpose(1, 2)?;

        let key_states = self.k_proj.forward(xs)?.reshape((
            b_size,
            seq_len,
            self.num_key_value_heads,
            self.head_dim,
        ))?;
        let key_states = self.k_layernorm.forward(&key_states)?.transpose(1, 2)?;

        let value_states = self
            .v_proj
            .forward(xs)?
            .reshape((b_size, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (query_states, key_states) =
            apply_rotary_pos_emb(&query_states, &key_states, cos, sin, false)?;
        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let attn_output = eager_attention_forward(
            &query_states,
            &key_states,
            &value_states,
            Some(self.num_key_value_groups),
            None,
            scale,
        )?;
        let attn_output = attn_output.reshape((b_size, seq_len, self.num_heads * self.head_dim))?;
        Ok(self.out_proj.forward(&attn_output)?)
    }
}

struct Lfm2_5EmbeddingShortConv {
    in_proj: Linear,
    conv: Conv1d,
    out_proj: Linear,
    hidden_size: usize,
    cache_len: usize,
}

impl Lfm2_5EmbeddingShortConv {
    fn new_from_vb(vb: VarBuilder, cfg: &Lfm2_5Config) -> Result<Self> {
        if cfg.conv_l_cache == 0 {
            return Err(anyhow!("conv_L_cache must be greater than zero"));
        }

        let hidden_size = cfg.hidden_size;
        let cache_len = cfg.conv_l_cache.saturating_sub(1);
        let in_proj = linear_no_bias(hidden_size, hidden_size * 3, vb.pp("in_proj"))?;
        let conv = get_conv1d(
            vb.pp("conv"),
            hidden_size,
            hidden_size,
            cfg.conv_l_cache,
            cache_len,
            1,
            1,
            hidden_size,
            cfg.conv_bias,
        )?;
        let out_proj = linear_no_bias(hidden_size, hidden_size, vb.pp("out_proj"))?;
        Ok(Self {
            in_proj,
            conv,
            out_proj,
            hidden_size,
            cache_len,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b_size, _seq_len, _) = xs.dims3()?;
        let mixed = self.in_proj.forward(xs)?.transpose(1, 2)?;
        let chunks = mixed.chunk(3, D::Minus2)?;
        let b = chunks[0].contiguous()?;
        let c = chunks[1].contiguous()?;
        let x = chunks[2].contiguous()?;
        let bx = b.broadcast_mul(&x)?;

        let conv_state = if self.cache_len == 0 {
            bx.clone()
        } else {
            let zeros = Tensor::zeros(
                (b_size, self.hidden_size, self.cache_len),
                bx.dtype(),
                bx.device(),
            )?;
            Tensor::cat(&[&zeros, &bx], D::Minus1)?
        };

        let conv_out = conv1d_depthwise(&conv_state, self.conv.weight(), self.conv.bias())?;
        let xs = c.broadcast_mul(&conv_out)?;
        let xs = xs.transpose(1, 2)?.contiguous()?;
        Ok(self.out_proj.forward(&xs)?)
    }
}

enum Lfm2_5EmbeddingOperator {
    Conv(Lfm2_5EmbeddingShortConv),
    FullAttention(Lfm2_5EmbeddingAttention),
}

struct Lfm2_5EmbeddingLayer {
    operator_norm: RmsNorm,
    operator: Lfm2_5EmbeddingOperator,
    ffn_norm: RmsNorm,
    feed_forward: Lfm2_5EmbeddingMLP,
}

impl Lfm2_5EmbeddingLayer {
    fn new_from_vb(
        vb: VarBuilder,
        cfg: &Lfm2_5Config,
        layer_kind: Lfm2_5LayerKind,
    ) -> Result<Self> {
        let operator_norm = rms_norm(cfg.hidden_size, cfg.norm_eps, vb.pp("operator_norm"))?;
        let ffn_norm = rms_norm(cfg.hidden_size, cfg.norm_eps, vb.pp("ffn_norm"))?;
        let feed_forward = Lfm2_5EmbeddingMLP::new_from_vb(vb.pp("feed_forward"), cfg)?;
        let operator = match layer_kind {
            Lfm2_5LayerKind::Conv => Lfm2_5EmbeddingOperator::Conv(
                Lfm2_5EmbeddingShortConv::new_from_vb(vb.pp("conv"), cfg)?,
            ),
            Lfm2_5LayerKind::FullAttention => Lfm2_5EmbeddingOperator::FullAttention(
                Lfm2_5EmbeddingAttention::new_from_vb(vb.pp("self_attn"), cfg)?,
            ),
        };

        Ok(Self {
            operator_norm,
            operator,
            ffn_norm,
            feed_forward,
        })
    }

    fn forward(&self, xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let residual = xs.clone();
        let xs = self.operator_norm.forward(xs)?;
        let xs = match &self.operator {
            Lfm2_5EmbeddingOperator::Conv(op) => op.forward(&xs)?,
            Lfm2_5EmbeddingOperator::FullAttention(op) => op.forward(&xs, cos, sin)?,
        };
        let xs = residual.add(&xs)?;
        let residual = xs.clone();
        let xs = self.ffn_norm.forward(&xs)?;
        let xs = self.feed_forward.forward(&xs)?;
        Ok(residual.add(&xs)?)
    }
}

pub struct Lfm2_5BidirectionalModel {
    embed_tokens: Embedding,
    layers: Vec<Lfm2_5EmbeddingLayer>,
    embedding_norm: RmsNorm,
    rotary_emb: RoPE,
}

impl Lfm2_5BidirectionalModel {
    pub fn new_from_vb(vb: VarBuilder, cfg: &Lfm2_5Config) -> Result<Self> {
        if cfg.hidden_size % cfg.num_attention_heads != 0 {
            return Err(anyhow!(
                "hidden_size {} is not divisible by num_attention_heads {}",
                cfg.hidden_size,
                cfg.num_attention_heads
            ));
        }
        if cfg.num_attention_heads % cfg.num_key_value_heads != 0 {
            return Err(anyhow!(
                "num_attention_heads {} is not divisible by num_key_value_heads {}",
                cfg.num_attention_heads,
                cfg.num_key_value_heads
            ));
        }
        let layer_kinds = validate_layer_types(&cfg.layer_types)?;
        if layer_kinds.len() != cfg.num_hidden_layers {
            return Err(anyhow!(
                "layer_types length {} does not match num_hidden_layers {}",
                layer_kinds.len(),
                cfg.num_hidden_layers
            ));
        }
        if cfg.conv_l_cache == 0 {
            return Err(anyhow!("conv_L_cache must be greater than zero"));
        }

        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;
        let embedding_norm = rms_norm(cfg.hidden_size, cfg.norm_eps, vb.pp("embedding_norm"))?;
        let rotary_emb = RoPE::new(cfg.head_dim(), cfg.rope_parameters.rope_theta, vb.device())?;
        let vb_layers = vb.pp("layers");
        let layers = layer_kinds
            .into_iter()
            .enumerate()
            .map(|(idx, layer_kind)| {
                Lfm2_5EmbeddingLayer::new_from_vb(vb_layers.pp(idx), cfg, layer_kind)
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            embed_tokens,
            layers,
            embedding_norm,
            rotary_emb,
        })
    }

    pub fn forward_hidden(&self, input_ids: &Tensor) -> Result<Tensor> {
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;
        let (_, seq_len, _) = hidden_states.dims3()?;
        let (cos, sin) = self
            .rotary_emb
            .forward(0, seq_len, hidden_states.device())?;
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, &cos, &sin)?;
        }
        Ok(self.embedding_norm.forward(&hidden_states)?)
    }
}

pub struct Lfm2_5EmbeddingSafetensorsBackend {
    tokenizer: TokenizerModel,
    model: Lfm2_5BidirectionalModel,
    device: Device,
    prompts: HashMap<String, String>,
    pooling: Lfm2_5EmbeddingPoolingStrategy,
    normalize: bool,
}

impl Lfm2_5EmbeddingSafetensorsBackend {
    pub fn load(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        if !std::path::Path::new(path).is_dir() {
            return Err(anyhow!("model dir not found: {}", path));
        }

        let cfg = Lfm2_5EmbeddingConfig::load(path)?;
        let tokenizer = TokenizerModel::init(path)?;
        let device = get_device(device);
        let dtype = get_dtype(dtype, cfg.base.dtype.as_str());
        let model_list = find_type_files(path, "safetensors")?;
        if model_list.is_empty() {
            return Err(anyhow!("no safetensors files found in {path}"));
        }
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, &device)? };
        let model = Lfm2_5BidirectionalModel::new_from_vb(vb, &cfg.base)?;

        Ok(Self {
            tokenizer,
            model,
            device,
            prompts: cfg.prompts,
            pooling: cfg.pooling,
            normalize: cfg.normalize,
        })
    }

    pub fn embed_texts(
        &mut self,
        input: &[String],
        options: EmbeddingOptions,
    ) -> Result<Vec<Vec<f32>>> {
        if input.is_empty() {
            return Err(anyhow!("embedding input cannot be empty"));
        }

        let mut output = Vec::with_capacity(input.len());
        for text in input {
            output.push(self.embed_one(text, options.prompt_name)?);
        }
        Ok(output)
    }

    fn prompt_for_name(&self, prompt_name: EmbeddingPromptName) -> Result<&str> {
        let key = match prompt_name {
            EmbeddingPromptName::Query => "query",
            EmbeddingPromptName::Document => "document",
        };
        self.prompts
            .get(key)
            .map(|value| value.as_str())
            .ok_or_else(|| anyhow!("missing {key} prompt mapping"))
    }

    fn embed_one(&self, text: &str, prompt_name: EmbeddingPromptName) -> Result<Vec<f32>> {
        let prompt = self.prompt_for_name(prompt_name)?;
        let rendered = format!("{prompt}{text}");
        let input_ids = self.tokenizer.text_encode(rendered, &self.device)?;
        let hidden = self
            .model
            .forward_hidden(&input_ids)?
            .squeeze(0)?
            .to_dtype(DType::F32)?;
        let mut pooled = match self.pooling {
            Lfm2_5EmbeddingPoolingStrategy::Cls => hidden.i(0)?.to_vec1::<f32>()?,
        };
        if self.normalize {
            l2_normalize(&mut pooled);
        }
        Ok(pooled)
    }
}
