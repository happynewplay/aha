use anyhow::{Result, anyhow};
use candle_core::Tensor;

#[cfg(feature = "onnx-runtime")]
use candle_core::{DType, IndexOp};

#[cfg(feature = "onnx-runtime")]
use std::path::{Path, PathBuf};

#[cfg(feature = "onnx-runtime")]
use half::f16;
#[cfg(feature = "onnx-runtime")]
use ndarray::{Array, IxDyn};

#[cfg(feature = "onnx-runtime")]
use crate::models::common::onnx::create_session;

#[cfg(feature = "onnx-runtime")]
pub struct GlmOcrOnnxCacheEntry {
    pub name: String,
    pub dims: Vec<i64>,
    pub data: Vec<f16>,
}

#[cfg(feature = "onnx-runtime")]
#[derive(Clone)]
struct OnnxInputDescriptor {
    name: String,
    shape: Vec<i64>,
    kind: Option<OnnxTensorKind>,
}

#[cfg(feature = "onnx-runtime")]
#[derive(Clone, Copy)]
enum OnnxTensorKind {
    Bool,
    I32,
    I64,
    F16,
    F32,
}

#[cfg(feature = "onnx-runtime")]
fn map_tensor_kind(ty: ort::value::TensorElementType) -> Option<OnnxTensorKind> {
    match ty {
        ort::value::TensorElementType::Bool => Some(OnnxTensorKind::Bool),
        ort::value::TensorElementType::Int32 => Some(OnnxTensorKind::I32),
        ort::value::TensorElementType::Int64 => Some(OnnxTensorKind::I64),
        ort::value::TensorElementType::Float16 => Some(OnnxTensorKind::F16),
        ort::value::TensorElementType::Float32 => Some(OnnxTensorKind::F32),
        _ => None,
    }
}

#[cfg(feature = "onnx-runtime")]
pub struct GlmOcrOnnxBackend {
    embed_session: ort::session::Session,
    decoder_session: ort::session::Session,
    vision_session: ort::session::Session,
    decoder_input_descriptors: Vec<OnnxInputDescriptor>,
    decoder_output_names: Vec<String>,
    vision_input_descriptors: Vec<OnnxInputDescriptor>,
    vision_output_names: Vec<String>,
    cache_values: Vec<GlmOcrOnnxCacheEntry>,
    spatial_merge_size: usize,
    next_mrope_pos: usize,
    prefill_seq_len: usize,
}

#[cfg(feature = "onnx-runtime")]
fn find_onnx_component_file(path: &str, marker: &str) -> Result<PathBuf> {
    let model_path = Path::new(path);
    if !model_path.exists() {
        return Err(anyhow!("onnx model path not found: {}", path));
    }

    if model_path.is_file() {
        let file_name = model_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default();
        if file_name.contains(marker) && file_name.ends_with(".onnx") {
            return Ok(model_path.to_path_buf());
        }
    }

    let search_root = if model_path.is_dir() {
        model_path.to_path_buf()
    } else {
        model_path
            .parent()
            .ok_or_else(|| anyhow!("onnx component parent directory not found for {}", path))?
            .to_path_buf()
    };

    let mut stack = vec![search_root];
    let mut matches = Vec::new();
    while let Some(current) = stack.pop() {
        for entry in std::fs::read_dir(&current)? {
            let entry = entry?;
            let entry_path = entry.path();
            if entry_path.is_dir() {
                stack.push(entry_path);
                continue;
            }
            let file_name = entry_path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default();
            if file_name.contains(marker) && file_name.ends_with(".onnx") {
                matches.push(entry_path);
            }
        }
    }
    matches.sort();
    matches.into_iter().next().ok_or_else(|| {
        anyhow!(
            "unable to locate onnx component {} under {}",
            marker,
            model_path.display()
        )
    })
}

#[cfg(feature = "onnx-runtime")]
impl GlmOcrOnnxBackend {
    pub fn load(onnx_path: &str, spatial_merge_size: usize) -> Result<Self> {
        let embed_file = find_onnx_component_file(onnx_path, "embed_tokens")?;
        let decoder_file = find_onnx_component_file(onnx_path, "decoder_model_merged")?;
        let vision_file = find_onnx_component_file(onnx_path, "vision_encoder")?;
        let embed_bundle = create_session(&embed_file.to_string_lossy(), None)?;
        let decoder_bundle = create_session(&decoder_file.to_string_lossy(), None)?;
        let vision_bundle = create_session(&vision_file.to_string_lossy(), None)?;

        let decoder_input_descriptors = decoder_bundle
            .session
            .inputs()
            .iter()
            .map(|input| {
                let (shape, kind) = match input.dtype() {
                    ort::value::ValueType::Tensor { ty, shape, .. } => (
                        shape.iter().copied().collect::<Vec<_>>(),
                        map_tensor_kind(*ty),
                    ),
                    _ => (Vec::new(), None),
                };
                OnnxInputDescriptor {
                    name: input.name().to_string(),
                    shape,
                    kind,
                }
            })
            .collect::<Vec<_>>();
        let vision_input_descriptors = vision_bundle
            .session
            .inputs()
            .iter()
            .map(|input| {
                let (shape, kind) = match input.dtype() {
                    ort::value::ValueType::Tensor { ty, shape, .. } => (
                        shape.iter().copied().collect::<Vec<_>>(),
                        map_tensor_kind(*ty),
                    ),
                    _ => (Vec::new(), None),
                };
                OnnxInputDescriptor {
                    name: input.name().to_string(),
                    shape,
                    kind,
                }
            })
            .collect::<Vec<_>>();

        Ok(Self {
            embed_session: embed_bundle.session,
            decoder_session: decoder_bundle.session,
            vision_session: vision_bundle.session,
            decoder_input_descriptors,
            decoder_output_names: decoder_bundle.output_names,
            vision_input_descriptors,
            vision_output_names: vision_bundle.output_names,
            cache_values: Vec::new(),
            spatial_merge_size,
            next_mrope_pos: 0,
            prefill_seq_len: 0,
        })
    }

    pub fn clear_cache(&mut self) {
        self.cache_values.clear();
        self.next_mrope_pos = 0;
        self.prefill_seq_len = 0;
    }

    pub fn forward_logits(
        &mut self,
        input_ids: &[u32],
        image_mask: Option<&Tensor>,
        pixel_values: Option<&Tensor>,
        image_grid_thw: Option<&Tensor>,
        position_start: usize,
    ) -> Result<Vec<f32>> {
        if input_ids.is_empty() {
            return Err(anyhow!("glm-ocr onnx input_ids cannot be empty"));
        }

        let (mut embed_data, hidden_size) = self.embed_input_ids(input_ids)?;
        if let Some(pixel_values) = pixel_values {
            let image_mask =
                image_mask.ok_or_else(|| anyhow!("glm-ocr onnx image_mask is required"))?;
            let image_grid_thw =
                image_grid_thw.ok_or_else(|| anyhow!("glm-ocr onnx image_grid_thw is required"))?;
            self.apply_vision_embeds(
                &mut embed_data,
                hidden_size,
                image_mask,
                pixel_values,
                image_grid_thw,
            )?;
        }

        let position_ids = if self.cache_values.is_empty() {
            if let (Some(mask), Some(grid_thw)) = (image_mask, image_grid_thw) {
                self.compute_prefill_position_ids(mask, grid_thw, input_ids.len())?
            } else {
                self.next_mrope_pos = input_ids.len();
                self.prefill_seq_len = input_ids.len();
                build_text_position_ids(0, input_ids.len())
            }
        } else {
            let decode_pos = self
                .next_mrope_pos
                .saturating_add(position_start.saturating_sub(self.prefill_seq_len));
            build_text_position_ids(decode_pos, input_ids.len())
        };

        let mut decoder_inputs = Vec::with_capacity(self.decoder_input_descriptors.len());
        for desc in &self.decoder_input_descriptors {
            let value = match desc.name.as_str() {
                "inputs_embeds" => build_float_input(
                    desc,
                    vec![1_i64, input_ids.len() as i64, hidden_size as i64],
                    &embed_data,
                )?,
                "attention_mask" => {
                    let attention_mask = self.build_attention_mask(input_ids.len())?;
                    build_i64_like_input(
                        desc,
                        vec![1_i64, attention_mask.len() as i64],
                        &attention_mask,
                    )?
                }
                "position_ids" => build_i64_like_input(
                    desc,
                    vec![3_i64, 1_i64, input_ids.len() as i64],
                    &position_ids,
                )?,
                "past_sequence_length" => build_i64_scalar_input(desc, self.past_seq_len() as i64)?,
                name if name.starts_with("past_key_values.") => {
                    if let Some(cache) = self.cache_values.iter().find(|entry| entry.name == name) {
                        ort::value::Tensor::from_array((cache.dims.clone(), cache.data.clone()))?
                            .into_dyn()
                    } else {
                        build_zero_cache_input(desc)?
                    }
                }
                _ => build_zero_input(desc)?,
            };
            decoder_inputs.push((desc.name.clone(), value));
        }

        let decoder_outputs = self.decoder_session.run(decoder_inputs)?;
        let logits_value = decoder_outputs
            .get("logits")
            .or_else(|| {
                self.decoder_output_names
                    .first()
                    .and_then(|name| decoder_outputs.get(name))
            })
            .ok_or_else(|| anyhow!("glm-ocr onnx output logits not found"))?;
        let logits = extract_last_logits(logits_value)?;

        let mut new_cache_values = Vec::new();
        for desc in &self.decoder_input_descriptors {
            let name = &desc.name;
            if !name.starts_with("past_key_values.") {
                continue;
            }
            let present_name = name.replace("past_key_values.", "present.");
            let value = decoder_outputs
                .get(&present_name)
                .ok_or_else(|| anyhow!("missing glm-ocr onnx output {}", present_name))?;
            let (shape, data) = value.try_extract_tensor::<f16>()?;
            new_cache_values.push(GlmOcrOnnxCacheEntry {
                name: name.clone(),
                dims: shape.iter().copied().collect::<Vec<_>>(),
                data: data.to_vec(),
            });
        }
        self.cache_values = new_cache_values;
        Ok(logits)
    }

    fn embed_input_ids(&mut self, input_ids: &[u32]) -> Result<(Vec<f32>, usize)> {
        let outputs = self.embed_session.run(vec![(
            "input_ids".to_string(),
            ort::value::Tensor::from_array((
                vec![1_i64, input_ids.len() as i64],
                input_ids.iter().map(|id| *id as i64).collect::<Vec<_>>(),
            ))?
            .into_dyn(),
        )])?;
        let embed_value = outputs
            .get("inputs_embeds")
            .ok_or_else(|| anyhow!("glm-ocr onnx output inputs_embeds not found"))?;

        if let Ok((shape, embed_data)) = embed_value.try_extract_tensor::<f32>() {
            if shape.len() != 3 || shape[0] != 1 {
                return Err(anyhow!(
                    "unexpected glm-ocr onnx inputs_embeds shape: {}",
                    shape
                ));
            }
            return Ok((embed_data.to_vec(), shape[2] as usize));
        }
        if let Ok((shape, embed_data)) = embed_value.try_extract_tensor::<f16>() {
            if shape.len() != 3 || shape[0] != 1 {
                return Err(anyhow!(
                    "unexpected glm-ocr onnx inputs_embeds shape: {}",
                    shape
                ));
            }
            return Ok((
                embed_data
                    .iter()
                    .map(|value| value.to_f32())
                    .collect::<Vec<_>>(),
                shape[2] as usize,
            ));
        }
        Err(anyhow!(
            "glm-ocr onnx inputs_embeds output must be a f32/f16 tensor"
        ))
    }

    fn apply_vision_embeds(
        &mut self,
        embed_data: &mut [f32],
        hidden_size: usize,
        image_mask: &Tensor,
        pixel_values: &Tensor,
        image_grid_thw: &Tensor,
    ) -> Result<()> {
        let (vision_embeds, vision_rows, vision_hidden) =
            self.run_vision_encoder(pixel_values, image_grid_thw)?;
        if vision_hidden != hidden_size {
            return Err(anyhow!(
                "glm-ocr onnx vision hidden size mismatch: vision={}, text={}",
                vision_hidden,
                hidden_size
            ));
        }

        let image_positions = image_mask
            .squeeze(0)?
            .to_dtype(DType::U8)?
            .to_vec1::<u8>()?
            .into_iter()
            .enumerate()
            .filter_map(|(idx, value)| (value == 1).then_some(idx))
            .collect::<Vec<_>>();

        if image_positions.len() != vision_rows {
            return Err(anyhow!(
                "glm-ocr onnx image token/vision embed mismatch: image_tokens={}, vision_embeds={}",
                image_positions.len(),
                vision_rows
            ));
        }

        for (row_idx, token_idx) in image_positions.into_iter().enumerate() {
            let src_start = row_idx * hidden_size;
            let dst_start = token_idx * hidden_size;
            let src_end = src_start + hidden_size;
            let dst_end = dst_start + hidden_size;
            embed_data[dst_start..dst_end].copy_from_slice(&vision_embeds[src_start..src_end]);
        }
        Ok(())
    }

    fn run_vision_encoder(
        &mut self,
        pixel_values: &Tensor,
        image_grid_thw: &Tensor,
    ) -> Result<(Vec<f32>, usize, usize)> {
        let raw_pixel_values_shape = pixel_values
            .dims()
            .iter()
            .map(|dim| *dim as i64)
            .collect::<Vec<_>>();
        let pixel_values_data = pixel_values
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        let raw_image_grid_shape = image_grid_thw
            .dims()
            .iter()
            .map(|dim| *dim as i64)
            .collect::<Vec<_>>();
        let image_grid_data = image_grid_thw
            .flatten_all()?
            .to_vec1::<u32>()?
            .into_iter()
            .map(|value| value as i64)
            .collect::<Vec<_>>();

        let mut vision_inputs = Vec::with_capacity(self.vision_input_descriptors.len());
        for desc in &self.vision_input_descriptors {
            let value = match desc.name.as_str() {
                "pixel_values" => {
                    let pixel_values_shape =
                        align_onnx_input_shape(desc, raw_pixel_values_shape.clone());
                    build_float_input(desc, pixel_values_shape, &pixel_values_data)?
                }
                "image_grid_thw" => {
                    let image_grid_shape =
                        align_onnx_input_shape(desc, raw_image_grid_shape.clone());
                    build_i64_like_input(desc, image_grid_shape, &image_grid_data)?
                }
                _ => build_zero_input(desc)?,
            };
            vision_inputs.push((desc.name.clone(), value));
        }

        let outputs = self.vision_session.run(vision_inputs)?;
        let output_value = outputs
            .get("image_embeds")
            .or_else(|| outputs.get("vision_embeds"))
            .or_else(|| {
                self.vision_output_names
                    .first()
                    .and_then(|name| outputs.get(name))
            })
            .ok_or_else(|| anyhow!("glm-ocr onnx vision output not found"))?;

        if let Ok((shape, values)) = output_value.try_extract_tensor::<f32>() {
            let shape_vec = shape.iter().copied().collect::<Vec<_>>();
            return extract_vision_output(shape_vec.as_slice(), values.to_vec());
        }
        if let Ok((shape, values)) = output_value.try_extract_tensor::<f16>() {
            let shape_vec = shape.iter().copied().collect::<Vec<_>>();
            let values = values
                .iter()
                .map(|value| value.to_f32())
                .collect::<Vec<_>>();
            return extract_vision_output(shape_vec.as_slice(), values);
        }
        Err(anyhow!(
            "glm-ocr onnx vision output must be a f32/f16 tensor"
        ))
    }

    fn past_seq_len(&self) -> usize {
        self.cache_values
            .iter()
            .find(|entry| entry.name.ends_with(".key"))
            .map(|entry| entry.dims.get(2).copied().unwrap_or_default() as usize)
            .unwrap_or(0)
    }

    fn build_attention_mask(&self, current_seq_len: usize) -> Result<Vec<i64>> {
        let total_len = self.past_seq_len().saturating_add(current_seq_len);
        if total_len == 0 {
            return Err(anyhow!("glm-ocr onnx attention length cannot be zero"));
        }
        Ok(vec![1_i64; total_len])
    }

    fn compute_prefill_position_ids(
        &mut self,
        image_mask: &Tensor,
        grid_thw: &Tensor,
        seq_len: usize,
    ) -> Result<Vec<i64>> {
        let t_dim = grid_thw.i(0)?.to_dtype(DType::F32)?.to_scalar::<f32>()? as usize;
        let h_dim = grid_thw.i(1)?.to_dtype(DType::F32)?.to_scalar::<f32>()? as usize;
        let w_dim = grid_thw.i(2)?.to_dtype(DType::F32)?.to_scalar::<f32>()? as usize;

        let llm_grid_t = t_dim;
        let llm_grid_h = h_dim / self.spatial_merge_size;
        let llm_grid_w = w_dim / self.spatial_merge_size;
        let num_image_tokens = llm_grid_t * llm_grid_h * llm_grid_w;

        let mask_vec = image_mask
            .squeeze(0)?
            .to_dtype(DType::U8)?
            .to_vec1::<u8>()?;
        let mut t_ids = Vec::with_capacity(seq_len);
        let mut h_ids = Vec::with_capacity(seq_len);
        let mut w_ids = Vec::with_capacity(seq_len);
        let mut st_idx: i64 = 0;
        let mut i = 0usize;

        while i < seq_len {
            let is_img = mask_vec[i] == 1;
            let start = i;
            while i < seq_len && (mask_vec[i] == 1) == is_img {
                i += 1;
            }
            let run_len = i - start;

            if is_img {
                if run_len != num_image_tokens {
                    return Err(anyhow!(
                        "glm-ocr onnx image token count mismatch: mask={}, grid={}",
                        run_len,
                        num_image_tokens
                    ));
                }
                for ti in 0..llm_grid_t {
                    for hi in 0..llm_grid_h {
                        for wi in 0..llm_grid_w {
                            t_ids.push(ti as i64 + st_idx);
                            h_ids.push(hi as i64 + st_idx);
                            w_ids.push(wi as i64 + st_idx);
                        }
                    }
                }
                let max_offset = (llm_grid_t as i64 - 1)
                    .max(llm_grid_h as i64 - 1)
                    .max(llm_grid_w as i64 - 1);
                st_idx += max_offset + 1;
            } else {
                for j in 0..run_len {
                    let pos = st_idx + j as i64;
                    t_ids.push(pos);
                    h_ids.push(pos);
                    w_ids.push(pos);
                }
                st_idx += run_len as i64;
            }
        }

        self.next_mrope_pos = st_idx as usize;
        self.prefill_seq_len = seq_len;

        let mut out = Vec::with_capacity(seq_len * 3);
        out.extend_from_slice(&t_ids);
        out.extend_from_slice(&h_ids);
        out.extend_from_slice(&w_ids);
        Ok(out)
    }
}

#[cfg(feature = "onnx-runtime")]
fn build_text_position_ids(position_start: usize, seq_len: usize) -> Vec<i64> {
    let positions = (position_start..position_start + seq_len)
        .map(|idx| idx as i64)
        .collect::<Vec<_>>();
    let mut ids = Vec::with_capacity(seq_len * 3);
    for _ in 0..3 {
        ids.extend_from_slice(&positions);
    }
    ids
}

#[cfg(feature = "onnx-runtime")]
fn align_onnx_input_shape(desc: &OnnxInputDescriptor, actual_shape: Vec<i64>) -> Vec<i64> {
    if desc.shape.is_empty() || desc.shape.len() <= actual_shape.len() {
        return actual_shape;
    }

    let mut aligned = actual_shape;
    while aligned.len() < desc.shape.len() {
        aligned.insert(0, 1);
    }
    aligned
}

#[cfg(feature = "onnx-runtime")]
fn build_i64_scalar_input(desc: &OnnxInputDescriptor, value: i64) -> Result<ort::value::DynValue> {
    match desc.kind {
        Some(OnnxTensorKind::I32) => {
            Ok(ort::value::Tensor::from_array((vec![1_i64], vec![value as i32]))?.into_dyn())
        }
        Some(OnnxTensorKind::I64) | None => {
            Ok(ort::value::Tensor::from_array((vec![1_i64], vec![value]))?.into_dyn())
        }
        _ => Err(anyhow!(
            "unsupported glm-ocr onnx scalar input dtype for {}",
            desc.name
        )),
    }
}

#[cfg(feature = "onnx-runtime")]
fn build_i64_like_input(
    desc: &OnnxInputDescriptor,
    shape: Vec<i64>,
    data: &[i64],
) -> Result<ort::value::DynValue> {
    match desc.kind {
        Some(OnnxTensorKind::I32) => Ok(ort::value::Tensor::from_array((
            shape,
            data.iter().map(|value| *value as i32).collect::<Vec<_>>(),
        ))?
        .into_dyn()),
        Some(OnnxTensorKind::I64) | None => {
            Ok(ort::value::Tensor::from_array((shape, data.to_vec()))?.into_dyn())
        }
        Some(OnnxTensorKind::Bool) => Ok(ort::value::Tensor::from_array((
            shape,
            data.iter().map(|value| *value != 0).collect::<Vec<_>>(),
        ))?
        .into_dyn()),
        _ => Err(anyhow!(
            "unsupported glm-ocr onnx integer-like input dtype for {}",
            desc.name
        )),
    }
}

#[cfg(feature = "onnx-runtime")]
fn build_float_input(
    desc: &OnnxInputDescriptor,
    shape: Vec<i64>,
    data: &[f32],
) -> Result<ort::value::DynValue> {
    match desc.kind {
        Some(OnnxTensorKind::F16) => Ok(ort::value::Tensor::from_array((
            shape,
            data.iter()
                .map(|value| f16::from_f32(*value))
                .collect::<Vec<_>>(),
        ))?
        .into_dyn()),
        Some(OnnxTensorKind::F32) | None => {
            Ok(ort::value::Tensor::from_array((shape, data.to_vec()))?.into_dyn())
        }
        _ => Err(anyhow!(
            "unsupported glm-ocr onnx float input dtype for {}",
            desc.name
        )),
    }
}

#[cfg(feature = "onnx-runtime")]
fn build_zero_cache_input(desc: &OnnxInputDescriptor) -> Result<ort::value::DynValue> {
    let shape = desc
        .shape
        .iter()
        .enumerate()
        .map(|(idx, dim)| {
            if *dim >= 0 {
                *dim
            } else if idx == 2 {
                0
            } else {
                1
            }
        })
        .collect::<Vec<_>>();
    build_zero_input_with_shape(desc, shape)
}

#[cfg(feature = "onnx-runtime")]
fn build_zero_input(desc: &OnnxInputDescriptor) -> Result<ort::value::DynValue> {
    let shape = desc
        .shape
        .iter()
        .map(|dim| if *dim < 0 { 1 } else { *dim })
        .collect::<Vec<_>>();
    build_zero_input_with_shape(desc, shape)
}

#[cfg(feature = "onnx-runtime")]
fn build_zero_input_with_shape(
    desc: &OnnxInputDescriptor,
    shape: Vec<i64>,
) -> Result<ort::value::DynValue> {
    let kind = desc
        .kind
        .ok_or_else(|| anyhow!("unsupported glm-ocr onnx input dtype for {}", desc.name))?;
    let elem_count = shape.iter().try_fold(1_usize, |acc, dim| {
        if *dim < 0 {
            Err(anyhow!(
                "cannot resolve dynamic onnx shape for input {}: {:?}",
                desc.name,
                shape
            ))
        } else {
            Ok(acc.saturating_mul(*dim as usize))
        }
    })?;
    let has_zero_dim = shape.contains(&0);
    let make_ndarray = || {
        let dims = shape.iter().map(|dim| *dim as usize).collect::<Vec<_>>();
        IxDyn(&dims)
    };
    let value = match kind {
        OnnxTensorKind::Bool => {
            if has_zero_dim {
                let arr = Array::from_shape_vec(make_ndarray(), vec![false; elem_count])?;
                ort::value::Tensor::from_array(arr)?.into_dyn()
            } else {
                ort::value::Tensor::from_array((shape, vec![false; elem_count]))?.into_dyn()
            }
        }
        OnnxTensorKind::I32 => {
            if has_zero_dim {
                let arr = Array::from_shape_vec(make_ndarray(), vec![0_i32; elem_count])?;
                ort::value::Tensor::from_array(arr)?.into_dyn()
            } else {
                ort::value::Tensor::from_array((shape, vec![0_i32; elem_count]))?.into_dyn()
            }
        }
        OnnxTensorKind::I64 => {
            if has_zero_dim {
                let arr = Array::from_shape_vec(make_ndarray(), vec![0_i64; elem_count])?;
                ort::value::Tensor::from_array(arr)?.into_dyn()
            } else {
                ort::value::Tensor::from_array((shape, vec![0_i64; elem_count]))?.into_dyn()
            }
        }
        OnnxTensorKind::F16 => {
            if has_zero_dim {
                let arr =
                    Array::from_shape_vec(make_ndarray(), vec![f16::from_f32(0.0); elem_count])?;
                ort::value::Tensor::from_array(arr)?.into_dyn()
            } else {
                ort::value::Tensor::from_array((shape, vec![f16::from_f32(0.0); elem_count]))?
                    .into_dyn()
            }
        }
        OnnxTensorKind::F32 => {
            if has_zero_dim {
                let arr = Array::from_shape_vec(make_ndarray(), vec![0_f32; elem_count])?;
                ort::value::Tensor::from_array(arr)?.into_dyn()
            } else {
                ort::value::Tensor::from_array((shape, vec![0_f32; elem_count]))?.into_dyn()
            }
        }
    };
    Ok(value)
}

#[cfg(feature = "onnx-runtime")]
fn extract_last_logits(value: &ort::value::DynValue) -> Result<Vec<f32>> {
    if let Ok((shape, values)) = value.try_extract_tensor::<f32>() {
        return extract_last_logits_from_shape(
            shape.iter().copied().collect::<Vec<_>>().as_slice(),
            values,
        );
    }
    if let Ok((shape, values)) = value.try_extract_tensor::<f16>() {
        let values = values
            .iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();
        return extract_last_logits_from_shape(
            shape.iter().copied().collect::<Vec<_>>().as_slice(),
            &values,
        );
    }
    Err(anyhow!(
        "glm-ocr onnx logits output must be a f32/f16 tensor"
    ))
}

#[cfg(feature = "onnx-runtime")]
fn extract_last_logits_from_shape(shape: &[i64], values: &[f32]) -> Result<Vec<f32>> {
    match shape {
        [1, seq_len, vocab_size] => {
            let seq_len = *seq_len as usize;
            let vocab_size = *vocab_size as usize;
            let start = seq_len
                .checked_sub(1)
                .ok_or_else(|| anyhow!("glm-ocr onnx logits sequence is empty"))?
                * vocab_size;
            Ok(values[start..start + vocab_size].to_vec())
        }
        [seq_len, vocab_size] => {
            let seq_len = *seq_len as usize;
            let vocab_size = *vocab_size as usize;
            let start = seq_len
                .checked_sub(1)
                .ok_or_else(|| anyhow!("glm-ocr onnx logits sequence is empty"))?
                * vocab_size;
            Ok(values[start..start + vocab_size].to_vec())
        }
        _ => Err(anyhow!("unexpected glm-ocr onnx logits shape: {:?}", shape)),
    }
}

#[cfg(feature = "onnx-runtime")]
fn extract_vision_output(shape: &[i64], values: Vec<f32>) -> Result<(Vec<f32>, usize, usize)> {
    match shape {
        [rows, hidden] => Ok((values, *rows as usize, *hidden as usize)),
        [1, rows, hidden] => Ok((values, *rows as usize, *hidden as usize)),
        _ => Err(anyhow!(
            "unexpected glm-ocr onnx vision output shape: {:?}",
            shape
        )),
    }
}

#[cfg(all(test, feature = "onnx-runtime"))]
mod tests {
    use super::{OnnxInputDescriptor, OnnxTensorKind, align_onnx_input_shape};

    #[test]
    fn align_onnx_input_shape_prepends_batch_dimension_when_descriptor_has_higher_rank() {
        let desc = OnnxInputDescriptor {
            name: "image_grid_thw".to_string(),
            shape: vec![-1, 3],
            kind: Some(OnnxTensorKind::I64),
        };
        assert_eq!(align_onnx_input_shape(&desc, vec![3]), vec![1, 3]);
    }
}

#[cfg(not(feature = "onnx-runtime"))]
pub struct GlmOcrOnnxBackend;

#[cfg(not(feature = "onnx-runtime"))]
impl GlmOcrOnnxBackend {
    pub fn load(_onnx_path: &str, _spatial_merge_size: usize) -> Result<Self> {
        Err(anyhow!(
            "onnx runtime support is not enabled; rebuild with --features onnx-runtime"
        ))
    }

    pub fn clear_cache(&mut self) {}

    pub fn forward_logits(
        &mut self,
        _input_ids: &[u32],
        _image_mask: Option<&Tensor>,
        _pixel_values: Option<&Tensor>,
        _image_grid_thw: Option<&Tensor>,
        _position_start: usize,
    ) -> Result<Vec<f32>> {
        Err(anyhow!(
            "onnx runtime support is not enabled; rebuild with --features onnx-runtime"
        ))
    }
}
