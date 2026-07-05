use serde::Deserialize;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct RopeParameters {
    pub rope_theta: f32,
    pub rope_type: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Lfm2_5Config {
    pub architectures: Vec<String>,
    pub model_type: String,
    pub dtype: String,
    pub block_auto_adjust_ff_dim: bool,
    pub block_dim: usize,
    pub block_ff_dim: usize,
    pub block_ffn_dim_multiplier: f32,
    pub block_mlp_init_scale: f32,
    pub block_multiple_of: usize,
    pub block_norm_eps: f64,
    pub block_out_init_scale: f32,
    pub block_use_swiglu: bool,
    pub block_use_xavier_init: bool,
    pub bos_token_id: u32,
    #[serde(rename = "conv_L_cache")]
    pub conv_l_cache: usize,
    pub conv_bias: bool,
    pub conv_dim: usize,
    pub conv_use_xavier_init: bool,
    pub eos_token_id: u32,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub layer_types: Vec<String>,
    pub max_position_embeddings: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub norm_eps: f64,
    pub pad_token_id: u32,
    pub rope_parameters: RopeParameters,
    pub tie_embedding: bool,
    pub use_cache: bool,
    pub use_pos_enc: bool,
    pub vocab_size: usize,
}

impl Lfm2_5Config {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn adjusted_intermediate_size(&self) -> usize {
        if !self.block_auto_adjust_ff_dim {
            return self.intermediate_size;
        }

        let scaled = (self.intermediate_size as f64)
            * (2.0_f64 / 3.0_f64)
            * self.block_ffn_dim_multiplier as f64;
        round_up_to_multiple(scaled.ceil() as usize, self.block_multiple_of)
    }
}

fn round_up_to_multiple(value: usize, multiple: usize) -> usize {
    if multiple == 0 {
        return value;
    }
    let rem = value % multiple;
    if rem == 0 {
        value
    } else {
        value + (multiple - rem)
    }
}

#[cfg(test)]
mod tests {
    use super::Lfm2_5Config;
    use anyhow::Result;

    #[test]
    fn deserialize_lfm2_5_config_from_local_shape() -> Result<()> {
        let cfg: Lfm2_5Config = serde_json::from_value(serde_json::json!({
            "architectures": ["Lfm2ForCausalLM"],
            "model_type": "lfm2",
            "dtype": "bfloat16",
            "block_auto_adjust_ff_dim": true,
            "block_dim": 1024,
            "block_ff_dim": 6656,
            "block_ffn_dim_multiplier": 1.0,
            "block_mlp_init_scale": 1.0,
            "block_multiple_of": 256,
            "block_norm_eps": 1e-5,
            "block_out_init_scale": 1.0,
            "block_use_swiglu": true,
            "block_use_xavier_init": true,
            "bos_token_id": 1,
            "conv_L_cache": 3,
            "conv_bias": false,
            "conv_dim": 1024,
            "conv_use_xavier_init": true,
            "eos_token_id": 7,
            "hidden_size": 1024,
            "intermediate_size": 6656,
            "layer_types": ["conv", "conv", "full_attention"],
            "max_position_embeddings": 128000,
            "num_hidden_layers": 16,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "norm_eps": 1e-5,
            "pad_token_id": 0,
            "rope_parameters": { "rope_theta": 1000000.0, "rope_type": "default" },
            "tie_embedding": true,
            "use_cache": true,
            "use_pos_enc": true,
            "vocab_size": 65536
        }))?;

        assert_eq!(cfg.model_type, "lfm2");
        assert_eq!(
            cfg.layer_types,
            vec![
                String::from("conv"),
                String::from("conv"),
                String::from("full_attention"),
            ]
        );
        assert_eq!(cfg.conv_l_cache, 3);
        assert_eq!(cfg.adjusted_intermediate_size(), 4608);
        Ok(())
    }
}
