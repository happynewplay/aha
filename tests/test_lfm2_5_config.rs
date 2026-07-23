use aha::models::lfm2_5::config::Lfm2_5Config;
use anyhow::Result;

#[test]
fn lfm2_5_config_accepts_both_tied_embedding_keys() -> Result<()> {
    let cfg: Lfm2_5Config = serde_json::from_value(serde_json::json!({
        "architectures": ["Lfm2ForCausalLM"],
        "model_type": "lfm2",
        "dtype": "bfloat16",
        "block_auto_adjust_ff_dim": false,
        "block_dim": 1024,
        "block_ff_dim": 2560,
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
        "intermediate_size": 2560,
        "layer_types": ["conv", "conv", "full_attention"],
        "max_position_embeddings": 128000,
        "num_hidden_layers": 3,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "norm_eps": 1e-5,
        "pad_token_id": 0,
        "rope_parameters": { "rope_theta": 1000000.0, "rope_type": "default" },
        "tie_embedding": true,
        "tie_word_embeddings": true,
        "use_cache": true,
        "use_pos_enc": true,
        "vocab_size": 65536
    }))?;

    assert!(cfg.tie_embedding);
    assert!(cfg.tied_embeddings());
    Ok(())
}
