use std::path::{Path, PathBuf};

use aha::models::{
    ArtifactKind, LoadSpec, ModelPaths, WhichModel,
    common::{onnx::ensure_ort_dylib_path, retrieval::cosine_similarity},
    mxbai_embed_xsmall_v1::generate::MxbaiEmbedXsmallV1Model,
};
use anyhow::{Context, Result, anyhow};
#[cfg(feature = "onnx-runtime")]
use ort::session::Session;

const MXBAI_EMBED_XSMALL_V1_DIR: &str = r"D:\model_download\mxbai-embed-xsmall-v1";
const MXBAI_EMBED_XSMALL_V1_GGUF_DIR: &str = r"D:\model_download\mxbai-embed-xsmall-v1\gguf";
const MXBAI_EMBED_XSMALL_V1_ONNX_DIR: &str = r"D:\model_download\mxbai-embed-xsmall-v1\onnx";

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

#[test]
fn mxbai_embed_xsmall_v1_safetensors_can_load() -> Result<()> {
    require_existing_dir(MXBAI_EMBED_XSMALL_V1_DIR)?;

    let _model = MxbaiEmbedXsmallV1Model::init(MXBAI_EMBED_XSMALL_V1_DIR, None, None)
        .with_context(|| {
            format!(
                "failed to init mxbai-embed-xsmall-v1 safetensors model from {}",
                MXBAI_EMBED_XSMALL_V1_DIR
            )
        })?;
    Ok(())
}

#[test]
fn mxbai_embed_xsmall_v1_gguf_file_can_load() -> Result<()> {
    let gguf_path = first_file_with_extension(MXBAI_EMBED_XSMALL_V1_GGUF_DIR, "gguf")?;
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
fn mxbai_embed_xsmall_v1_onnx_file_can_load() -> Result<()> {
    let onnx_path = first_file_with_extension_recursive(MXBAI_EMBED_XSMALL_V1_ONNX_DIR, "onnx")?;
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
fn mxbai_embed_xsmall_v1_onnxruntime_can_create_session() -> Result<()> {
    let onnx_path = first_file_with_extension_recursive(MXBAI_EMBED_XSMALL_V1_ONNX_DIR, "onnx")?;

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
fn mxbai_embed_xsmall_v1_safetensors_init_from_spec_can_embed() -> Result<()> {
    require_existing_dir(MXBAI_EMBED_XSMALL_V1_DIR)?;

    let spec = LoadSpec {
        model: WhichModel::MxbaiEmbedXsmallV1,
        artifact: ArtifactKind::Safetensors,
        paths: ModelPaths {
            weight_dir: Some(MXBAI_EMBED_XSMALL_V1_DIR.to_string()),
            ..Default::default()
        },
    };

    let mut model = MxbaiEmbedXsmallV1Model::init_from_spec(&spec, None, None)?;
    let output = model.embed(&["test safetensors embedding".to_string()])?;
    assert_eq!(output.len(), 1);
    assert_eq!(output[0].len(), 384);
    Ok(())
}

#[test]
fn mxbai_embed_xsmall_v1_gguf_init_from_spec_can_embed() -> Result<()> {
    let gguf_path = match first_file_with_extension(MXBAI_EMBED_XSMALL_V1_GGUF_DIR, "gguf") {
        Ok(path) => path,
        Err(err) => {
            println!("skip gguf init_from_spec test: {err}");
            return Ok(());
        }
    };

    let spec = LoadSpec {
        model: WhichModel::MxbaiEmbedXsmallV1,
        artifact: ArtifactKind::Gguf,
        paths: ModelPaths {
            gguf_path: Some(gguf_path.to_string_lossy().to_string()),
            tokenizer_dir: Some(MXBAI_EMBED_XSMALL_V1_DIR.to_string()),
            ..Default::default()
        },
    };

    let mut model = MxbaiEmbedXsmallV1Model::init_from_spec(&spec, None, None)?;
    let output = model.embed(&["test gguf embedding".to_string()])?;
    assert_eq!(output.len(), 1);
    assert_eq!(output[0].len(), 384);
    Ok(())
}

#[test]
fn mxbai_embed_xsmall_v1_onnx_init_from_spec_can_embed() -> Result<()> {
    if let Err(err) = ensure_ort_dylib_path() {
        println!("skip onnx init test: {err}");
        return Ok(());
    }

    let spec = LoadSpec {
        model: WhichModel::MxbaiEmbedXsmallV1,
        artifact: ArtifactKind::Onnx,
        paths: ModelPaths {
            onnx_path: Some(MXBAI_EMBED_XSMALL_V1_ONNX_DIR.to_string()),
            tokenizer_dir: Some(MXBAI_EMBED_XSMALL_V1_DIR.to_string()),
            ..Default::default()
        },
    };

    let mut model = MxbaiEmbedXsmallV1Model::init_from_spec(&spec, None, None)?;
    let output = model.embed(&["test onnx embedding".to_string()])?;
    assert_eq!(output.len(), 1);
    assert_eq!(output[0].len(), 384);
    Ok(())
}

#[test]
fn mxbai_embed_xsmall_v1_native_and_gguf_embeddings_are_close() -> Result<()> {
    let gguf_path = match first_file_with_extension(MXBAI_EMBED_XSMALL_V1_GGUF_DIR, "gguf") {
        Ok(path) => path,
        Err(err) => {
            println!("skip native/gguf similarity test: {err}");
            return Ok(());
        }
    };

    let text = "Rust provides strong ownership guarantees for concurrent systems.";
    let mut native_model = MxbaiEmbedXsmallV1Model::init(MXBAI_EMBED_XSMALL_V1_DIR, None, None)?;
    let mut gguf_model = MxbaiEmbedXsmallV1Model::init_gguf(
        &gguf_path.to_string_lossy(),
        Some(MXBAI_EMBED_XSMALL_V1_DIR),
        None,
        None,
    )?;

    let native_embedding = native_model.embed(&[text.to_string()])?;
    let gguf_embedding = gguf_model.embed(&[text.to_string()])?;

    let similarity = cosine_similarity(&native_embedding[0], &gguf_embedding[0])?;
    assert!(
        similarity > 0.98,
        "native/gguf embedding similarity too low: {similarity}"
    );
    Ok(())
}

#[test]
fn mxbai_embed_xsmall_v1_native_and_onnx_embeddings_are_close() -> Result<()> {
    if let Err(err) = ensure_ort_dylib_path() {
        println!("skip cross-backend similarity test: {err}");
        return Ok(());
    }

    let text = "Rust provides strong ownership guarantees for concurrent systems.";
    let mut native_model = MxbaiEmbedXsmallV1Model::init(MXBAI_EMBED_XSMALL_V1_DIR, None, None)?;
    let mut onnx_model = MxbaiEmbedXsmallV1Model::init_onnx(
        MXBAI_EMBED_XSMALL_V1_ONNX_DIR,
        Some(MXBAI_EMBED_XSMALL_V1_DIR),
    )?;

    let native_embedding = native_model.embed(&[text.to_string()])?;
    let onnx_embedding = onnx_model.embed(&[text.to_string()])?;

    let similarity = cosine_similarity(&native_embedding[0], &onnx_embedding[0])?;
    assert!(
        similarity > 0.98,
        "native/onnx embedding similarity too low: {similarity}"
    );
    Ok(())
}
