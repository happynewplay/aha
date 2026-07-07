use std::path::Path;

use anyhow::Result;
use candle_transformers::models::bert::Config as BertConfig;
use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MxbaiEmbedXsmallV1PoolingStrategy {
    Cls,
    Mean,
    Max,
    MeanSqrtLen,
}

#[derive(Debug, Clone)]
pub struct MxbaiEmbedXsmallV1Config {
    pub base: BertConfig,
    pub pooling: MxbaiEmbedXsmallV1PoolingStrategy,
    pub normalize: bool,
    pub max_seq_length: usize,
    pub do_lower_case: bool,
}

#[derive(Debug, Deserialize, Default)]
struct TokenizerConfig {
    #[serde(default)]
    do_lower_case: bool,
    max_length: Option<usize>,
    model_max_length: Option<usize>,
}

#[derive(Debug, Deserialize, Default)]
struct AngleConfig {
    max_length: Option<usize>,
    pooling_strategy: Option<String>,
}

#[derive(Debug, Deserialize)]
struct PoolingConfig {
    #[serde(default)]
    pooling_mode_cls_token: bool,
    #[serde(default)]
    pooling_mode_mean_tokens: bool,
    #[serde(default)]
    pooling_mode_max_tokens: bool,
    #[serde(default)]
    pooling_mode_mean_sqrt_len_tokens: bool,
}

#[derive(Debug, Deserialize)]
struct ModuleEntry {
    #[serde(default)]
    r#type: String,
}

fn default_max_seq_length() -> usize {
    512
}

fn pooling_from_angle_config(value: Option<&str>) -> Option<MxbaiEmbedXsmallV1PoolingStrategy> {
    match value {
        Some("cls") => Some(MxbaiEmbedXsmallV1PoolingStrategy::Cls),
        Some("avg") | Some("mean") => Some(MxbaiEmbedXsmallV1PoolingStrategy::Mean),
        Some("max") => Some(MxbaiEmbedXsmallV1PoolingStrategy::Max),
        Some("mean_sqrt_len") => Some(MxbaiEmbedXsmallV1PoolingStrategy::MeanSqrtLen),
        _ => None,
    }
}

impl MxbaiEmbedXsmallV1Config {
    pub fn load(path: &str) -> Result<Self> {
        let config_path = Path::new(path).join("config.json");
        let base: BertConfig = serde_json::from_slice(&std::fs::read(config_path)?)?;

        let tokenizer_config_path = Path::new(path).join("tokenizer_config.json");
        let tokenizer_config = if tokenizer_config_path.exists() {
            serde_json::from_slice::<TokenizerConfig>(&std::fs::read(tokenizer_config_path)?)?
        } else {
            TokenizerConfig::default()
        };

        let angle_config_path = Path::new(path).join("angle_config.json");
        let angle_config = if angle_config_path.exists() {
            serde_json::from_slice::<AngleConfig>(&std::fs::read(angle_config_path)?)?
        } else {
            AngleConfig::default()
        };

        let pooling_path = Path::new(path).join("1_Pooling").join("config.json");
        let pooling = if pooling_path.exists() {
            let cfg: PoolingConfig = serde_json::from_slice(&std::fs::read(pooling_path)?)?;
            if cfg.pooling_mode_mean_tokens {
                MxbaiEmbedXsmallV1PoolingStrategy::Mean
            } else if cfg.pooling_mode_cls_token {
                MxbaiEmbedXsmallV1PoolingStrategy::Cls
            } else if cfg.pooling_mode_max_tokens {
                MxbaiEmbedXsmallV1PoolingStrategy::Max
            } else if cfg.pooling_mode_mean_sqrt_len_tokens {
                MxbaiEmbedXsmallV1PoolingStrategy::MeanSqrtLen
            } else {
                pooling_from_angle_config(angle_config.pooling_strategy.as_deref())
                    .unwrap_or(MxbaiEmbedXsmallV1PoolingStrategy::Mean)
            }
        } else {
            pooling_from_angle_config(angle_config.pooling_strategy.as_deref())
                .unwrap_or(MxbaiEmbedXsmallV1PoolingStrategy::Mean)
        };

        let modules_path = Path::new(path).join("modules.json");
        let normalize = if modules_path.exists() {
            let modules: Vec<ModuleEntry> = serde_json::from_slice(&std::fs::read(modules_path)?)?;
            modules
                .iter()
                .any(|module| module.r#type.ends_with(".Normalize"))
        } else {
            false
        };

        let max_seq_length = angle_config
            .max_length
            .or(tokenizer_config.model_max_length)
            .or(tokenizer_config.max_length)
            .unwrap_or_else(default_max_seq_length);

        Ok(Self {
            base,
            pooling,
            normalize,
            max_seq_length,
            do_lower_case: tokenizer_config.do_lower_case,
        })
    }
}
