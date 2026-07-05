use anyhow::{Result, anyhow};
use candle_core::{D, Tensor};
use candle_nn::{
    Conv1d, Embedding, Linear, Module, RmsNorm, VarBuilder, embedding, linear_no_bias, rms_norm,
};

use crate::{
    models::common::{conv1d_depthwise, eager_attention_forward, get_conv1d},
    position_embed::rope::{RoPE, apply_rotary_pos_emb},
};

use super::config::Lfm2_5Config;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lfm2_5LayerKind {
    Conv,
    FullAttention,
}

pub fn validate_layer_types(layer_types: &[String]) -> Result<Vec<Lfm2_5LayerKind>> {
    layer_types
        .iter()
        .map(|layer| match layer.as_str() {
            "conv" => Ok(Lfm2_5LayerKind::Conv),
            "full_attention" => Ok(Lfm2_5LayerKind::FullAttention),
            other => Err(anyhow!("unsupported lfm2.5 layer type: {other}")),
        })
        .collect()
}

struct Lfm2_5MLP {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl Lfm2_5MLP {
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

struct Lfm2_5Attention {
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
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Lfm2_5Attention {
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
            kv_cache: None,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
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
        let (key_states, value_states) = match &self.kv_cache {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => {
                let key_states = Tensor::cat(&[prev_k, &key_states], 2)?;
                let value_states = Tensor::cat(&[prev_v, &value_states], 2)?;
                (key_states, value_states)
            }
        };

        self.kv_cache = Some((key_states.clone(), value_states.clone()));
        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let attn_output = eager_attention_forward(
            &query_states,
            &key_states,
            &value_states,
            Some(self.num_key_value_groups),
            attention_mask,
            scale,
        )?;
        let attn_output = attn_output.reshape((b_size, seq_len, self.num_heads * self.head_dim))?;
        Ok(self.out_proj.forward(&attn_output)?)
    }

    fn clear_cache(&mut self) {
        self.kv_cache = None;
    }
}

struct Lfm2_5ShortConv {
    in_proj: Linear,
    conv: Conv1d,
    out_proj: Linear,
    hidden_size: usize,
    cache_len: usize,
    conv_state_cache: Option<Tensor>,
}

impl Lfm2_5ShortConv {
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
            conv_state_cache: None,
        })
    }

    fn forward(&mut self, xs: &Tensor) -> Result<Tensor> {
        let (b_size, _seq_len, _) = xs.dims3()?;
        let mixed = self.in_proj.forward(xs)?.transpose(1, 2)?;
        let chunks = mixed.chunk(3, D::Minus2)?;
        let b = chunks[0].contiguous()?;
        let c = chunks[1].contiguous()?;
        let x = chunks[2].contiguous()?;
        let bx = b.broadcast_mul(&x)?;

        let conv_state = match &self.conv_state_cache {
            Some(cache) => Tensor::cat(&[cache, &bx], D::Minus1)?,
            None => {
                if self.cache_len == 0 {
                    bx.clone()
                } else {
                    let zeros = Tensor::zeros(
                        (b_size, self.hidden_size, self.cache_len),
                        bx.dtype(),
                        bx.device(),
                    )?;
                    Tensor::cat(&[&zeros, &bx], D::Minus1)?
                }
            }
        };

        let conv_out = conv1d_depthwise(&conv_state, self.conv.weight(), self.conv.bias())?;
        if self.cache_len > 0 {
            let state_len = conv_state.dim(D::Minus1)?;
            self.conv_state_cache =
                Some(conv_state.narrow(D::Minus1, state_len - self.cache_len, self.cache_len)?);
        } else {
            self.conv_state_cache = None;
        }

        let xs = c.broadcast_mul(&conv_out)?;
        let xs = xs.transpose(1, 2)?.contiguous()?;
        let xs = self.out_proj.forward(&xs)?;
        Ok(xs)
    }

    fn clear_cache(&mut self) {
        self.conv_state_cache = None;
    }
}

enum Lfm2_5Operator {
    Conv(Lfm2_5ShortConv),
    FullAttention(Lfm2_5Attention),
}

struct Lfm2_5DecoderLayer {
    operator_norm: RmsNorm,
    operator: Lfm2_5Operator,
    ffn_norm: RmsNorm,
    feed_forward: Lfm2_5MLP,
}

impl Lfm2_5DecoderLayer {
    fn new_from_vb(
        vb: VarBuilder,
        cfg: &Lfm2_5Config,
        layer_kind: Lfm2_5LayerKind,
    ) -> Result<Self> {
        let operator_norm = rms_norm(cfg.hidden_size, cfg.norm_eps, vb.pp("operator_norm"))?;
        let ffn_norm = rms_norm(cfg.hidden_size, cfg.norm_eps, vb.pp("ffn_norm"))?;
        let feed_forward = Lfm2_5MLP::new_from_vb(vb.pp("feed_forward"), cfg)?;
        let operator = match layer_kind {
            Lfm2_5LayerKind::Conv => {
                Lfm2_5Operator::Conv(Lfm2_5ShortConv::new_from_vb(vb.pp("conv"), cfg)?)
            }
            Lfm2_5LayerKind::FullAttention => Lfm2_5Operator::FullAttention(
                Lfm2_5Attention::new_from_vb(vb.pp("self_attn"), cfg)?,
            ),
        };

        Ok(Self {
            operator_norm,
            operator,
            ffn_norm,
            feed_forward,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = xs.clone();
        let xs = self.operator_norm.forward(xs)?;
        let xs = match &mut self.operator {
            Lfm2_5Operator::Conv(op) => op.forward(&xs)?,
            Lfm2_5Operator::FullAttention(op) => op.forward(&xs, cos, sin, attention_mask)?,
        };
        let xs = residual.add(&xs)?;
        let residual = xs.clone();
        let xs = self.ffn_norm.forward(&xs)?;
        let xs = self.feed_forward.forward(&xs)?;
        Ok(residual.add(&xs)?)
    }

    fn clear_cache(&mut self) {
        match &mut self.operator {
            Lfm2_5Operator::Conv(op) => op.clear_cache(),
            Lfm2_5Operator::FullAttention(op) => op.clear_cache(),
        }
    }
}

pub struct Lfm2_5TextModel {
    embed_tokens: Embedding,
    layers: Vec<Lfm2_5DecoderLayer>,
    embedding_norm: RmsNorm,
    rotary_emb: RoPE,
}

impl Lfm2_5TextModel {
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
                Lfm2_5DecoderLayer::new_from_vb(vb_layers.pp(idx), cfg, layer_kind)
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            embed_tokens,
            layers,
            embedding_norm,
            rotary_emb,
        })
    }

    fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;
        let (batch_size, seq_len, _) = hidden_states.dims3()?;
        let (cos, sin) = self
            .rotary_emb
            .forward(seqlen_offset, seq_len, hidden_states.device())?;
        let attention_mask = if seq_len > 1 {
            Some(crate::utils::tensor_utils::prepare_causal_attention_mask(
                batch_size,
                seq_len,
                seqlen_offset,
                hidden_states.device(),
            )?)
        } else {
            None
        };

        for layer in self.layers.iter_mut() {
            hidden_states = layer.forward(&hidden_states, &cos, &sin, attention_mask.as_ref())?;
        }
        Ok(self.embedding_norm.forward(&hidden_states)?)
    }

    fn clear_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_cache();
        }
    }
}

pub struct Lfm2_5Model {
    language_model: Lfm2_5TextModel,
    lm_head: Linear,
}

impl Lfm2_5Model {
    pub fn new_from_vb(vb: VarBuilder, cfg: &Lfm2_5Config) -> Result<Self> {
        let language_model = Lfm2_5TextModel::new_from_vb(vb.pp("model"), cfg)?;
        let lm_head = if cfg.tie_embedding {
            Linear::new(language_model.embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };
        Ok(Self {
            language_model,
            lm_head,
        })
    }

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let hidden_states = self.language_model.forward(input_ids, seqlen_offset)?;
        let seq_len = hidden_states.dim(1)?;
        let hidden_state = hidden_states.narrow(1, seq_len.saturating_sub(1), 1)?;
        Ok(self.lm_head.forward(&hidden_state)?)
    }

    pub fn clear_cache(&mut self) {
        self.language_model.clear_cache();
    }
}

#[cfg(test)]
mod tests {
    use super::{Lfm2_5LayerKind, validate_layer_types};
    use anyhow::Result;

    #[test]
    fn validate_layer_types_accepts_conv_and_full_attention_only() -> Result<()> {
        let kinds = validate_layer_types(&["conv".to_string(), "full_attention".to_string()])?;
        assert!(matches!(kinds[0], Lfm2_5LayerKind::Conv));
        assert!(matches!(kinds[1], Lfm2_5LayerKind::FullAttention));
        Ok(())
    }
}
