use std::{
    collections::HashMap,
    io::{Read, Seek},
    path::Path,
};

use anyhow::{Result, anyhow};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Activation, Module, VarBuilder};

use crate::models::{
    common::{
        LlamaForCausalLM,
        gguf::{Gguf, load_gguf_file},
    },
    minicpm5::config::MiniCPM5Config,
};

pub(super) fn resolve_minicpm5_gguf_file(path: &str) -> Result<String> {
    let model_path = Path::new(path);
    if !model_path.exists() {
        return Err(anyhow!("gguf model path not found: {}", path));
    }

    if model_path.is_file() {
        if model_path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
        {
            return Ok(model_path.to_string_lossy().to_string());
        }
        return Err(anyhow!(
            "gguf model path does not point to a .gguf file: {}",
            path
        ));
    }

    let mut stack = vec![model_path.to_path_buf()];
    let mut matches = Vec::new();
    while let Some(current) = stack.pop() {
        for entry in std::fs::read_dir(&current)? {
            let entry = entry?;
            let candidate = entry.path();
            if candidate.is_dir() {
                stack.push(candidate);
                continue;
            }
            if candidate
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
            {
                matches.push(candidate);
            }
        }
    }

    matches.sort();
    matches
        .into_iter()
        .next()
        .map(|path| path.to_string_lossy().to_string())
        .ok_or_else(|| anyhow!("no .gguf file found in {}", model_path.display()))
}

pub struct MiniCPM5Model {
    model: LlamaForCausalLM,
}

impl MiniCPM5Model {
    pub fn new_from_vb(vb: VarBuilder, cfg: &MiniCPM5Config) -> Result<Self> {
        let model = LlamaForCausalLM::new(
            vb,
            cfg.vocab_size,
            cfg.hidden_size,
            cfg.num_hidden_layers,
            cfg.num_attention_heads,
            Some(cfg.num_key_value_heads),
            Some(cfg.head_dim),
            false,
            "self_attn",
            Some("o_proj"),
            cfg.intermediate_size,
            cfg.hidden_act.clone(),
            false,
            "mlp",
            cfg.rms_norm_eps,
            "input_layernorm",
            "post_attention_layernorm",
            cfg.rope_theta,
        )?;
        Ok(Self { model })
    }

    pub fn new_from_gguf(model_file: &str, device: &Device, dtype: DType) -> Result<Self> {
        let model_file = resolve_minicpm5_gguf_file(model_file)?;
        let mut gguf = load_gguf_file(&model_file, device)?;
        let arch = required_string(&gguf, "general.architecture")?;
        if arch != "llama" {
            return Err(anyhow!("unsupported gguf architecture: {arch}"));
        }

        let block_count = required_u32(&gguf, &format!("{arch}.block_count"))?;
        let num_heads = required_u32(&gguf, &format!("{arch}.attention.head_count"))?;
        let num_kv_heads = required_u32(&gguf, &format!("{arch}.attention.head_count_kv"))?;
        let head_dim = required_u32(&gguf, &format!("{arch}.attention.key_length"))?;
        let vocab_size = required_u32(&gguf, &format!("{arch}.vocab_size"))?;
        let hidden_size = required_u32(&gguf, &format!("{arch}.embedding_length"))?;
        let intermediate_size = required_u32(&gguf, &format!("{arch}.feed_forward_length"))?;
        let rms_norm_eps =
            required_f32(&gguf, &format!("{arch}.attention.layer_norm_rms_epsilon"))? as f64;
        let rope_theta = required_f32(&gguf, &format!("{arch}.rope.freq_base"))?;

        let tensors = load_llama_gguf_tensors(&mut gguf, block_count, device, dtype)?;
        let vb = VarBuilder::from_tensors(tensors, dtype, device);
        let model = LlamaForCausalLM::new(
            vb,
            vocab_size,
            hidden_size,
            block_count,
            num_heads,
            Some(num_kv_heads),
            Some(head_dim),
            false,
            "self_attn",
            Some("o_proj"),
            intermediate_size,
            Activation::Silu,
            false,
            "mlp",
            rms_norm_eps,
            "input_layernorm",
            "post_attention_layernorm",
            rope_theta,
        )?;
        Ok(Self { model })
    }

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let inputs_embeds = self.model.model.embed_tokens.forward(input_ids)?;
        self.model.forward(&inputs_embeds, seqlen_offset)
    }

    pub fn clear_kv_cache(&mut self) {
        self.model.clear_kv_cache();
    }
}

fn required_u32<R: Read + Seek>(gguf: &Gguf<R>, key: &str) -> Result<usize> {
    Ok(gguf
        .get_matedata(key)
        .map_err(|_| anyhow!("missing gguf metadata key: {key}"))?
        .to_u32()? as usize)
}

fn required_f32<R: Read + Seek>(gguf: &Gguf<R>, key: &str) -> Result<f32> {
    Ok(gguf
        .get_matedata(key)
        .map_err(|_| anyhow!("missing gguf metadata key: {key}"))?
        .to_f32()?)
}

fn required_string<R: Read + Seek>(gguf: &Gguf<R>, key: &str) -> Result<String> {
    Ok(gguf
        .get_matedata(key)
        .map_err(|_| anyhow!("missing gguf metadata key: {key}"))?
        .to_string()?
        .clone())
}

fn load_llama_gguf_tensors<R: Read + Seek>(
    gguf: &mut Gguf<R>,
    block_count: usize,
    device: &Device,
    dtype: DType,
) -> Result<HashMap<String, Tensor>> {
    let mut tensors = HashMap::new();

    insert_gguf_tensor(
        gguf,
        &mut tensors,
        "token_embd.weight",
        "model.embed_tokens.weight",
        device,
        dtype,
    )?;
    insert_gguf_tensor(
        gguf,
        &mut tensors,
        "output_norm.weight",
        "model.norm.weight",
        device,
        dtype,
    )?;

    let lm_head_source = if gguf.has_tensor("output.weight") {
        "output.weight"
    } else {
        "token_embd.weight"
    };
    insert_gguf_tensor(
        gguf,
        &mut tensors,
        lm_head_source,
        "lm_head.weight",
        device,
        dtype,
    )?;

    for layer_idx in 0..block_count {
        let prefix = format!("blk.{layer_idx}");
        let llama_prefix = format!("model.layers.{layer_idx}");

        insert_gguf_tensor(
            gguf,
            &mut tensors,
            &format!("{prefix}.attn_q.weight"),
            &format!("{llama_prefix}.self_attn.q_proj.weight"),
            device,
            dtype,
        )?;
        insert_gguf_tensor(
            gguf,
            &mut tensors,
            &format!("{prefix}.attn_k.weight"),
            &format!("{llama_prefix}.self_attn.k_proj.weight"),
            device,
            dtype,
        )?;
        insert_gguf_tensor(
            gguf,
            &mut tensors,
            &format!("{prefix}.attn_v.weight"),
            &format!("{llama_prefix}.self_attn.v_proj.weight"),
            device,
            dtype,
        )?;
        insert_gguf_tensor(
            gguf,
            &mut tensors,
            &format!("{prefix}.attn_output.weight"),
            &format!("{llama_prefix}.self_attn.o_proj.weight"),
            device,
            dtype,
        )?;
        insert_gguf_tensor(
            gguf,
            &mut tensors,
            &format!("{prefix}.attn_norm.weight"),
            &format!("{llama_prefix}.input_layernorm.weight"),
            device,
            dtype,
        )?;
        insert_gguf_tensor(
            gguf,
            &mut tensors,
            &format!("{prefix}.ffn_gate.weight"),
            &format!("{llama_prefix}.mlp.gate_proj.weight"),
            device,
            dtype,
        )?;
        insert_gguf_tensor(
            gguf,
            &mut tensors,
            &format!("{prefix}.ffn_up.weight"),
            &format!("{llama_prefix}.mlp.up_proj.weight"),
            device,
            dtype,
        )?;
        insert_gguf_tensor(
            gguf,
            &mut tensors,
            &format!("{prefix}.ffn_down.weight"),
            &format!("{llama_prefix}.mlp.down_proj.weight"),
            device,
            dtype,
        )?;
        insert_gguf_tensor(
            gguf,
            &mut tensors,
            &format!("{prefix}.ffn_norm.weight"),
            &format!("{llama_prefix}.post_attention_layernorm.weight"),
            device,
            dtype,
        )?;
    }

    Ok(tensors)
}

fn insert_gguf_tensor<R: Read + Seek>(
    gguf: &mut Gguf<R>,
    tensors: &mut HashMap<String, Tensor>,
    gguf_name: &str,
    tensor_name: &str,
    device: &Device,
    dtype: DType,
) -> Result<()> {
    let tensor = gguf
        .get_dequantized(gguf_name)
        .map_err(|err| anyhow!("failed to load gguf tensor {gguf_name}: {err}"))?
        .to_dtype(dtype)?;
    let tensor = tensor.to_device(device)?;
    tensors.insert(tensor_name.to_string(), tensor);
    Ok(())
}
