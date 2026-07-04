use candle_nn::Activation;

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct MiniCPM5Config {
    pub bos_token_id: u32,
    pub eos_token_id: Vec<u32>,
    pub pad_token_id: Option<u32>,
    pub hidden_act: Activation,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub torch_dtype: String,
    pub tie_word_embeddings: bool,
    pub use_cache: bool,
    pub vocab_size: usize,
}

impl MiniCPM5Config {
    pub fn stop_tokens(&self) -> &[u32] {
        &self.eos_token_id
    }
}

#[cfg(test)]
mod tests {
    use super::MiniCPM5Config;
    use candle_nn::Activation;

    #[test]
    fn minicpm5_config_deserializes_llama_style_fields() {
        let cfg: MiniCPM5Config = serde_json::from_value(serde_json::json!({
            "bos_token_id": 1,
            "eos_token_id": [1, 130073],
            "pad_token_id": 0,
            "hidden_act": "silu",
            "hidden_size": 1536,
            "intermediate_size": 4608,
            "max_position_embeddings": 32768,
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "rms_norm_eps": 1e-6,
            "rope_theta": 5000000.0,
            "torch_dtype": "bfloat16",
            "tie_word_embeddings": false,
            "use_cache": true,
            "vocab_size": 130560
        }))
        .unwrap();

        assert_eq!(cfg.hidden_size, 1536);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.eos_token_id, vec![1, 130073]);
        assert_eq!(cfg.hidden_act, Activation::Silu);
    }
}
