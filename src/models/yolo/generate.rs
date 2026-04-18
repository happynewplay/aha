use aha_openai_dive::v1::resources::chat::{
    ChatCompletionChunkResponse, ChatCompletionParameters, ChatCompletionResponse,
};
use anyhow::{Result, anyhow};
use rocket::futures::{Stream, stream};

use crate::models::LoadSpec;

use super::config::YoloConfig;
use super::model::{YoloModel, YoloPredictOptions, YoloResults};

/// YOLO vision model adapter. Note: YOLO is a vision model and does NOT
/// support the `GenerateModel` chat-completion trait. It is registered
/// as a `GenerateModel` only because the model registry requires a uniform
/// interface. Calling `generate()` or `generate_stream()` will return an error.
pub struct YoloGenerateModel {
    backend: YoloModel,
}

impl YoloGenerateModel {
    pub fn init_from_spec(spec: &LoadSpec) -> Result<Self> {
        Self::init_with_config(spec, YoloConfig::default())
    }

    pub fn init_with_config(spec: &LoadSpec, config: YoloConfig) -> Result<Self> {
        Ok(Self {
            backend: YoloModel::init_with_config(spec, config)?,
        })
    }

    pub fn predict(&mut self, source: &str) -> Result<Vec<YoloResults>> {
        self.backend.predict(source)
    }

    pub fn predict_with_options(
        &mut self,
        source: &str,
        options: &YoloPredictOptions,
    ) -> Result<Vec<YoloResults>> {
        self.backend.predict_with_options(source, options)
    }

    pub fn predict_stream_with_options<F>(
        &mut self,
        source: &str,
        options: &YoloPredictOptions,
        on_result: F,
    ) -> Result<()>
    where
        F: FnMut(&YoloResults) -> Result<bool>,
    {
        self.backend
            .predict_stream_with_options(source, options, on_result)
    }

    pub fn results_to_json(results: &[YoloResults]) -> Result<String> {
        YoloModel::results_to_json(results)
    }

    pub fn results_to_coco_json(results: &[YoloResults]) -> Result<String> {
        YoloModel::results_to_coco_json(results)
    }

    /// YOLO is a vision model, not a language model. This method always returns an error.
    /// Use `predict()` or `predict_with_options()` instead for object detection / classification.
    pub fn generate(&mut self, _mes: ChatCompletionParameters) -> Result<ChatCompletionResponse> {
        Err(anyhow!(
            "YOLO is a vision model and does not support chat completions; \
             use predict() or predict_with_options() for inference"
        ))
    }

    /// YOLO is a vision model, not a language model. This method always returns an error.
    /// Use `predict_stream_with_options()` instead for streaming video/frame inference.
    pub fn generate_stream(
        &mut self,
        _mes: ChatCompletionParameters,
    ) -> Result<
        Box<
            dyn Stream<Item = Result<ChatCompletionChunkResponse, anyhow::Error>>
                + Send
                + Unpin
                + '_,
        >,
    > {
        let error_stream = stream::once(async {
            Err(anyhow!(
                "YOLO is a vision model and does not support streaming chat completions; \
                 use predict_stream_with_options() for streaming inference"
            )) as Result<ChatCompletionChunkResponse, anyhow::Error>
        });
        Ok(Box::new(Box::pin(error_stream)))
    }
}

/// GenerateModel trait implementation for the model registry.
/// YOLO does not support chat completions; both methods return descriptive errors.
impl crate::models::GenerateModel for YoloGenerateModel {
    fn generate(&mut self, mes: ChatCompletionParameters) -> Result<ChatCompletionResponse> {
        YoloGenerateModel::generate(self, mes)
    }

    fn generate_stream(
        &mut self,
        mes: ChatCompletionParameters,
    ) -> Result<
        Box<
            dyn Stream<Item = Result<ChatCompletionChunkResponse, anyhow::Error>>
                + Send
                + Unpin
                + '_,
        >,
    > {
        YoloGenerateModel::generate_stream(self, mes)
    }
}
