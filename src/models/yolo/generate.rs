use aha_openai_dive::v1::resources::chat::{
    ChatCompletionChunkResponse, ChatCompletionParameters, ChatCompletionResponse,
};
use anyhow::{Result, anyhow};
use rocket::futures::{Stream, stream};

use crate::models::{GenerateModel, LoadSpec};

use super::model::{YoloModel, YoloPredictOptions, YoloResults};

pub struct YoloGenerateModel {
    backend: YoloModel,
}

impl YoloGenerateModel {
    pub fn init_from_spec(spec: &LoadSpec) -> Result<Self> {
        Ok(Self {
            backend: YoloModel::init_from_spec(spec)?,
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
}

impl GenerateModel for YoloGenerateModel {
    fn generate(&mut self, _mes: ChatCompletionParameters) -> Result<ChatCompletionResponse> {
        Err(anyhow!("yolo does not support chat completions"))
    }

    fn generate_stream(
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
            Err(anyhow!("yolo does not support streaming chat completions"))
                as Result<ChatCompletionChunkResponse, anyhow::Error>
        });
        Ok(Box::new(Box::pin(error_stream)))
    }
}
