use std::path::Path;

use aha::models::{
    ArtifactKind, EmbeddingOptions, EmbeddingPromptName, LoadSpec, ModelPaths, WhichModel,
    common::retrieval::cosine_similarity, lfm2_5_embedding::generate::Lfm2_5EmbeddingModel,
};
use anyhow::Result;

const DEFAULT_LFM2_5_EMBEDDING_DIR: &str = r"D:\model_download\LFM2.5-Embedding-350M";

fn resolve_lfm2_5_embedding_dir() -> Option<String> {
    let path = std::env::var("AHA_LFM2_5_EMBEDDING_350M_DIR")
        .unwrap_or_else(|_| DEFAULT_LFM2_5_EMBEDDING_DIR.to_string());
    if Path::new(&path).is_dir() {
        Some(path)
    } else {
        None
    }
}

#[test]
fn lfm2_5_embedding_safetensors_can_load() -> Result<()> {
    let Some(weight_dir) = resolve_lfm2_5_embedding_dir() else {
        println!(
            "skip lfm2.5 embedding test: dir not found, set AHA_LFM2_5_EMBEDDING_350M_DIR to run"
        );
        return Ok(());
    };

    let _model = Lfm2_5EmbeddingModel::init(&weight_dir, None, None)?;
    Ok(())
}

#[test]
fn lfm2_5_embedding_safetensors_init_from_spec_can_embed() -> Result<()> {
    let Some(weight_dir) = resolve_lfm2_5_embedding_dir() else {
        println!(
            "skip lfm2.5 embedding test: dir not found, set AHA_LFM2_5_EMBEDDING_350M_DIR to run"
        );
        return Ok(());
    };

    let spec = LoadSpec {
        model: WhichModel::LFM2_5Embedding350M,
        artifact: ArtifactKind::Safetensors,
        paths: ModelPaths {
            weight_dir: Some(weight_dir),
            ..Default::default()
        },
    };

    let mut model = Lfm2_5EmbeddingModel::init_from_spec(&spec, None, None)?;
    let output = model.embed_with_options(
        &["Rust embedding smoke test".to_string()],
        EmbeddingOptions::default(),
    )?;
    assert_eq!(output.len(), 1);
    assert_eq!(output[0].len(), 1024);
    Ok(())
}

#[test]
fn lfm2_5_embedding_query_and_document_prompts_differ() -> Result<()> {
    let Some(weight_dir) = resolve_lfm2_5_embedding_dir() else {
        println!(
            "skip lfm2.5 embedding test: dir not found, set AHA_LFM2_5_EMBEDDING_350M_DIR to run"
        );
        return Ok(());
    };

    let spec = LoadSpec {
        model: WhichModel::LFM2_5Embedding350M,
        artifact: ArtifactKind::Safetensors,
        paths: ModelPaths {
            weight_dir: Some(weight_dir),
            ..Default::default()
        },
    };

    let mut model = Lfm2_5EmbeddingModel::init_from_spec(&spec, None, None)?;
    let query = model.embed_with_options(
        &["What is the capital of France?".to_string()],
        EmbeddingOptions {
            prompt_name: EmbeddingPromptName::Query,
        },
    )?;
    let document = model.embed_with_options(
        &["What is the capital of France?".to_string()],
        EmbeddingOptions {
            prompt_name: EmbeddingPromptName::Document,
        },
    )?;
    assert_ne!(query[0], document[0]);
    Ok(())
}

#[test]
fn lfm2_5_embedding_query_matches_relevant_document() -> Result<()> {
    let Some(weight_dir) = resolve_lfm2_5_embedding_dir() else {
        println!(
            "skip lfm2.5 embedding test: dir not found, set AHA_LFM2_5_EMBEDDING_350M_DIR to run"
        );
        return Ok(());
    };

    let spec = LoadSpec {
        model: WhichModel::LFM2_5Embedding350M,
        artifact: ArtifactKind::Safetensors,
        paths: ModelPaths {
            weight_dir: Some(weight_dir),
            ..Default::default()
        },
    };

    let mut model = Lfm2_5EmbeddingModel::init_from_spec(&spec, None, None)?;
    let query = model.embed_with_options(
        &["What is the capital of France?".to_string()],
        EmbeddingOptions {
            prompt_name: EmbeddingPromptName::Query,
        },
    )?;
    let docs = model.embed_with_options(
        &[
            "Paris is the capital of France.".to_string(),
            "Rust is a systems programming language.".to_string(),
        ],
        EmbeddingOptions {
            prompt_name: EmbeddingPromptName::Document,
        },
    )?;

    let positive = cosine_similarity(&query[0], &docs[0])?;
    let negative = cosine_similarity(&query[0], &docs[1])?;
    assert!(positive > negative);
    Ok(())
}
