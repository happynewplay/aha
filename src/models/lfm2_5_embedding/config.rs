use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use anyhow::{Result, anyhow};
use serde::{Deserialize, de::DeserializeOwned};

use crate::models::lfm2_5::config::Lfm2_5Config;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lfm2_5EmbeddingPoolingStrategy {
    Cls,
}

#[derive(Debug, Clone)]
pub struct Lfm2_5EmbeddingConfig {
    pub base: Lfm2_5Config,
    pub prompts: HashMap<String, String>,
    pub pooling: Lfm2_5EmbeddingPoolingStrategy,
    pub normalize: bool,
    pub word_embedding_dimension: usize,
}

#[derive(Debug, Deserialize)]
struct SentenceTransformersConfig {
    #[serde(default)]
    prompts: HashMap<String, String>,
    similarity_fn_name: Option<String>,
}

#[derive(Debug, Deserialize)]
struct PoolingConfig {
    #[serde(default)]
    word_embedding_dimension: usize,
    #[serde(default)]
    pooling_mode_cls_token: bool,
    #[serde(default)]
    pooling_mode_mean_tokens: bool,
    #[serde(default)]
    pooling_mode_max_tokens: bool,
    #[serde(default)]
    pooling_mode_mean_sqrt_len_tokens: bool,
    #[serde(default)]
    pooling_mode_weightedmean_tokens: bool,
    #[serde(default)]
    pooling_mode_lasttoken: bool,
    #[serde(default)]
    include_prompt: bool,
}

#[derive(Debug, Deserialize)]
struct ModuleEntry {
    #[serde(default)]
    path: String,
    #[serde(default)]
    r#type: String,
}

pub fn resolve_prompts(mut prompts: HashMap<String, String>) -> Result<HashMap<String, String>> {
    let query = prompts.remove("query");
    let document = prompts.remove("document");
    match (query, document) {
        (Some(query), Some(document)) => Ok(HashMap::from([
            ("query".to_string(), query),
            ("document".to_string(), document),
        ])),
        (None, _) => Err(anyhow!(
            "missing query prompt in config_sentence_transformers.json"
        )),
        (_, None) => Err(anyhow!(
            "missing document prompt in config_sentence_transformers.json"
        )),
    }
}

pub fn resolve_pooling(
    cls: bool,
    mean: bool,
    max: bool,
    mean_sqrt_len: bool,
    weighted_mean: bool,
    last_token: bool,
) -> Result<Lfm2_5EmbeddingPoolingStrategy> {
    if cls && !mean && !max && !mean_sqrt_len && !weighted_mean && !last_token {
        return Ok(Lfm2_5EmbeddingPoolingStrategy::Cls);
    }
    Err(anyhow!(
        "LFM2.5-Embedding-350M currently supports only CLS pooling"
    ))
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let raw =
        std::fs::read(path).map_err(|err| anyhow!("failed to read {}: {}", path.display(), err))?;
    serde_json::from_slice(&raw)
        .map_err(|err| anyhow!("failed to parse {}: {}", path.display(), err))
}

fn validate_modules(modules: &[ModuleEntry]) -> Result<()> {
    let has_transformer = modules
        .iter()
        .any(|module| module.r#type.contains("Transformer"));
    if !has_transformer {
        return Err(anyhow!(
            "modules.json does not include a transformer module"
        ));
    }

    let has_pooling = modules
        .iter()
        .any(|module| module.path == "1_Pooling" || module.r#type.contains("Pooling"));
    if !has_pooling {
        return Err(anyhow!("modules.json does not include a 1_Pooling module"));
    }

    Ok(())
}

impl Lfm2_5EmbeddingConfig {
    pub fn normalize_from_similarity(value: Option<&str>) -> bool {
        matches!(value, Some("cosine"))
    }

    pub fn load(path: &str) -> Result<Self> {
        let root = PathBuf::from(path);

        let base_path = root.join("config.json");
        let base: Lfm2_5Config = read_json(&base_path)?;
        if !base
            .architectures
            .iter()
            .any(|name| name == "Lfm2BidirectionalModel")
        {
            return Err(anyhow!(
                "config.json does not declare Lfm2BidirectionalModel"
            ));
        }

        let st_path = root.join("config_sentence_transformers.json");
        let sentence_transformers: SentenceTransformersConfig = read_json(&st_path)?;
        let prompts = resolve_prompts(sentence_transformers.prompts)?;

        let modules_path = root.join("modules.json");
        let modules: Vec<ModuleEntry> = read_json(&modules_path)?;
        validate_modules(&modules)?;

        let pooling_path = root.join("1_Pooling").join("config.json");
        let pooling_cfg: PoolingConfig = read_json(&pooling_path)?;
        let pooling = resolve_pooling(
            pooling_cfg.pooling_mode_cls_token,
            pooling_cfg.pooling_mode_mean_tokens,
            pooling_cfg.pooling_mode_max_tokens,
            pooling_cfg.pooling_mode_mean_sqrt_len_tokens,
            pooling_cfg.pooling_mode_weightedmean_tokens,
            pooling_cfg.pooling_mode_lasttoken,
        )?;

        if pooling_cfg.word_embedding_dimension != base.hidden_size {
            return Err(anyhow!(
                "pooling word_embedding_dimension {} does not match hidden_size {}",
                pooling_cfg.word_embedding_dimension,
                base.hidden_size
            ));
        }

        Ok(Self {
            base,
            prompts,
            pooling,
            normalize: Self::normalize_from_similarity(
                sentence_transformers.similarity_fn_name.as_deref(),
            ),
            word_embedding_dimension: pooling_cfg.word_embedding_dimension,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{Lfm2_5EmbeddingConfig, resolve_pooling, resolve_prompts};
    use std::collections::HashMap;

    #[test]
    fn resolve_prompts_requires_query_and_document() {
        let mut prompts = HashMap::new();
        prompts.insert("query".to_string(), "query: ".to_string());
        let err = resolve_prompts(prompts).unwrap_err().to_string();
        assert!(err.contains("document"));
    }

    #[test]
    fn resolve_pooling_rejects_non_cls_layout() {
        let err = resolve_pooling(false, true, false, false, false, false)
            .unwrap_err()
            .to_string();
        assert!(err.contains("CLS"));
    }

    #[test]
    fn normalize_defaults_to_true_for_cosine_similarity() {
        assert!(Lfm2_5EmbeddingConfig::normalize_from_similarity(Some(
            "cosine"
        )));
        assert!(!Lfm2_5EmbeddingConfig::normalize_from_similarity(Some(
            "dot"
        )));
        assert!(!Lfm2_5EmbeddingConfig::normalize_from_similarity(None));
    }
}
