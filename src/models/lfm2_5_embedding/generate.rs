use anyhow::Result;
use candle_core::{DType, Device};

use crate::models::{
    EmbeddingOptions,
    artifact::{ArtifactKind, LoadSpec},
    lfm2_5_embedding::model::Lfm2_5EmbeddingSafetensorsBackend,
};

pub struct Lfm2_5EmbeddingModel {
    backend: Lfm2_5EmbeddingSafetensorsBackend,
}

impl Lfm2_5EmbeddingModel {
    pub fn init_from_spec(
        spec: &LoadSpec,
        device: Option<&Device>,
        dtype: Option<DType>,
    ) -> Result<Self> {
        match spec.resolved_artifact() {
            ArtifactKind::Safetensors => {
                let path = spec.paths.weight_dir.as_deref().ok_or_else(|| {
                    anyhow::anyhow!("weight_path is required for lfm2.5-embedding-350m safetensors")
                })?;
                Self::init(path, device, dtype)
            }
            ArtifactKind::Gguf => Err(anyhow::anyhow!(
                "lfm2.5-embedding-350m gguf runtime is not implemented yet"
            )),
            ArtifactKind::Onnx => Err(anyhow::anyhow!(
                "lfm2.5-embedding-350m onnx runtime is not implemented yet"
            )),
            ArtifactKind::Auto => unreachable!("artifact kind should be resolved before init"),
        }
    }

    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let backend = Lfm2_5EmbeddingSafetensorsBackend::load(path, device, dtype)?;
        Ok(Self { backend })
    }

    pub fn embed(&mut self, input: &[String]) -> Result<Vec<Vec<f32>>> {
        self.embed_with_options(input, EmbeddingOptions::default())
    }

    pub fn embed_with_options(
        &mut self,
        input: &[String],
        options: EmbeddingOptions,
    ) -> Result<Vec<Vec<f32>>> {
        self.backend.embed_texts(input, options)
    }
}
