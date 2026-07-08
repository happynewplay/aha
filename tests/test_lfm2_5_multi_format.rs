use std::path::{Path, PathBuf};

use aha::models::{
    ArtifactKind, GenerateModel, LoadSpec, ModelPaths, WhichModel,
    common::onnx::ensure_ort_dylib_path, lfm2_5::generate::Lfm2_5GenerateModel,
};
use aha_openai_dive::v1::resources::chat::ChatCompletionParameters;
use anyhow::{Context, Result, anyhow};
#[cfg(feature = "onnx-runtime")]
use ort::session::Session;

const LFM2_5_SAFETENSORS_DIR: &str = r"D:\model_download\LFM2.5-350M";
const LFM2_5_GGUF_DIR: &str = r"D:\model_download\LFM2.5-350M-GGUF";
const LFM2_5_ONNX_DIR: &str = r"D:\model_download\LFM2.5-350M-ONNX";

fn require_existing_dir(path: &str) -> Result<()> {
    let dir = Path::new(path);
    if !dir.exists() {
        return Err(anyhow!("model dir not found: {}", path));
    }
    if !dir.is_dir() {
        return Err(anyhow!("path is not a directory: {}", path));
    }
    Ok(())
}

fn first_file_with_extension(dir: &str, extension: &str) -> Result<PathBuf> {
    require_existing_dir(dir)?;

    let mut matches = std::fs::read_dir(dir)?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case(extension))
        })
        .collect::<Vec<_>>();

    matches.sort();
    matches
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("no .{} file found in {}", extension, dir))
}

fn first_file_with_extension_recursive(dir: &str, extension: &str) -> Result<PathBuf> {
    require_existing_dir(dir)?;

    let mut stack = vec![PathBuf::from(dir)];
    let mut matches = Vec::new();
    while let Some(current) = stack.pop() {
        for entry in std::fs::read_dir(&current)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if path
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case(extension))
            {
                matches.push(path);
            }
        }
    }

    matches.sort();
    matches
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("no .{} file found (recursive) in {}", extension, dir))
}

fn build_text_request() -> Result<ChatCompletionParameters> {
    Ok(serde_json::from_value(serde_json::json!({
        "model": "lfm2.5-350m",
        "max_tokens": 8,
        "messages": [
            {
                "role": "user",
                "content": "请用一句话介绍 Rust。"
            }
        ]
    }))?)
}

#[test]
fn lfm2_5_gguf_file_can_load() -> Result<()> {
    let gguf_path = first_file_with_extension(LFM2_5_GGUF_DIR, "gguf")?;
    let metadata = std::fs::metadata(&gguf_path)
        .with_context(|| format!("failed to read gguf metadata: {}", gguf_path.display()))?;
    if metadata.len() == 0 {
        return Err(anyhow!("gguf file is empty: {}", gguf_path.display()));
    }
    let _bytes = std::fs::read(&gguf_path)
        .with_context(|| format!("failed to read gguf file: {}", gguf_path.display()))?;
    Ok(())
}

#[test]
fn lfm2_5_onnx_file_can_load() -> Result<()> {
    let onnx_path = first_file_with_extension_recursive(LFM2_5_ONNX_DIR, "onnx")?;
    let metadata = std::fs::metadata(&onnx_path)
        .with_context(|| format!("failed to read onnx metadata: {}", onnx_path.display()))?;
    if metadata.len() == 0 {
        return Err(anyhow!("onnx file is empty: {}", onnx_path.display()));
    }
    let _bytes = std::fs::read(&onnx_path)
        .with_context(|| format!("failed to read onnx file: {}", onnx_path.display()))?;
    Ok(())
}

#[cfg(feature = "onnx-runtime")]
#[test]
fn lfm2_5_onnxruntime_can_create_session() -> Result<()> {
    let onnx_path = first_file_with_extension_recursive(LFM2_5_ONNX_DIR, "onnx")?;

    if let Err(err) = ensure_ort_dylib_path() {
        println!("skip onnxruntime session test: {err}");
        return Ok(());
    }

    let session = Session::builder()
        .context("failed to create onnxruntime session builder")?
        .commit_from_file(&onnx_path)
        .with_context(|| {
            format!(
                "failed to create onnxruntime session from {}",
                onnx_path.display()
            )
        })?;

    if session.inputs().is_empty() {
        return Err(anyhow!("onnxruntime session has no inputs"));
    }
    if session.outputs().is_empty() {
        return Err(anyhow!("onnxruntime session has no outputs"));
    }

    Ok(())
}

#[test]
fn lfm2_5_gguf_init_from_spec_can_generate() -> Result<()> {
    let gguf_path = match first_file_with_extension(LFM2_5_GGUF_DIR, "gguf") {
        Ok(path) => path,
        Err(err) => {
            println!("skip gguf init_from_spec test: {err}");
            return Ok(());
        }
    };

    let spec = LoadSpec {
        model: WhichModel::LFM2_5_350M,
        artifact: ArtifactKind::Gguf,
        paths: ModelPaths {
            gguf_path: Some(gguf_path.to_string_lossy().to_string()),
            tokenizer_dir: Some(LFM2_5_SAFETENSORS_DIR.to_string()),
            ..Default::default()
        },
    };

    let mut model = Lfm2_5GenerateModel::init_from_spec(&spec, None, None)?;
    let response = model.generate(build_text_request()?)?;
    let value = serde_json::to_value(response)?;
    let choices_len = value
        .get("choices")
        .and_then(|choices| choices.as_array())
        .map_or(0, |choices| choices.len());
    assert!(choices_len > 0, "expected at least one generated choice");
    Ok(())
}

#[test]
fn lfm2_5_onnx_init_from_spec_can_generate() -> Result<()> {
    if let Err(err) = ensure_ort_dylib_path() {
        println!("skip onnx init test: {err}");
        return Ok(());
    }

    let spec = LoadSpec {
        model: WhichModel::LFM2_5_350M,
        artifact: ArtifactKind::Onnx,
        paths: ModelPaths {
            onnx_path: Some(LFM2_5_ONNX_DIR.to_string()),
            tokenizer_dir: Some(LFM2_5_ONNX_DIR.to_string()),
            ..Default::default()
        },
    };

    let mut model = Lfm2_5GenerateModel::init_from_spec(&spec, None, None)?;
    let response = model.generate(build_text_request()?)?;
    let value = serde_json::to_value(response)?;
    let choices_len = value
        .get("choices")
        .and_then(|choices| choices.as_array())
        .map_or(0, |choices| choices.len());
    assert!(choices_len > 0, "expected at least one generated choice");
    Ok(())
}
