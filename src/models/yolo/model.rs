#[cfg(feature = "ffmpeg")]
use std::sync::OnceLock;
#[cfg(feature = "ffmpeg")]
use std::sync::atomic::Ordering;
use std::{
    path::Path,
    sync::{Arc, atomic::AtomicBool},
    thread,
    time::Instant,
};

use anyhow::{Result, anyhow};
#[cfg(feature = "ffmpeg")]
use ffmpeg_next as ffmpeg;
use font8x8::{BASIC_FONTS, UnicodeFonts};
use glob::glob;
use image::{DynamicImage, Rgba, RgbaImage, imageops};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[cfg(feature = "onnx-runtime")]
use crate::models::common::onnx::create_session;
use crate::{
    models::{ArtifactKind, LoadSpec},
    utils::{get_file_path, img_utils::load_image_from_url},
};

use super::config::YoloConfig;

const SUPPORTED_IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"];
const SUPPORTED_VIDEO_EXTENSIONS: &[&str] = &["mp4", "avi", "mov", "mkv", "webm"];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YoloBox {
    pub xyxy: [f32; 4],
    pub xywh: [f32; 4],
    pub conf: f32,
    pub cls: usize,
    pub label: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct YoloSpeed {
    pub preprocess_ms: f32,
    pub inference_ms: f32,
    pub postprocess_ms: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct YoloResults {
    pub boxes: Vec<YoloBox>,
    pub path: String,
    pub names: Vec<String>,
    pub speed: YoloSpeed,
    pub width: u32,
    pub height: u32,
    #[serde(skip_serializing)]
    pub orig_img: Option<DynamicImage>,
}

impl YoloResults {
    pub fn plot(&self) -> Result<RgbaImage> {
        let mut canvas = self
            .orig_img
            .as_ref()
            .ok_or_else(|| anyhow!("plot is not available for this result source"))?
            .to_rgba8();
        for detection in &self.boxes {
            draw_box(&mut canvas, detection, Rgba([255, 64, 64, 255]));
        }
        Ok(canvas)
    }

    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    pub fn latency_ms(&self) -> f32 {
        self.speed.preprocess_ms + self.speed.inference_ms + self.speed.postprocess_ms
    }

    pub fn save_txt(&self, path: &str) -> Result<()> {
        let content = self
            .boxes
            .iter()
            .map(|detection| {
                format!(
                    "{} {:.6} {:.6} {:.6} {:.6} {:.6}",
                    detection.cls,
                    detection.xywh[0],
                    detection.xywh[1],
                    detection.xywh[2],
                    detection.xywh[3],
                    detection.conf,
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(path, content)?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct YoloPredictOptions {
    pub stream: bool,
    pub workers: usize,
    pub batch_size: usize,
    pub max_frames: Option<usize>,
    pub frame_stride: usize,
    pub stop_flag: Option<Arc<AtomicBool>>,
}

impl Default for YoloPredictOptions {
    fn default() -> Self {
        Self {
            stream: false,
            workers: 1,
            batch_size: 16,
            max_frames: None,
            frame_stride: 1,
            stop_flag: None,
        }
    }
}

#[derive(Debug, Clone)]
struct ResolvedImageSource {
    source: String,
    image: DynamicImage,
}

#[derive(Debug, Clone)]
struct LetterboxMeta {
    scale: f32,
    pad_x: f32,
    pad_y: f32,
    orig_w: u32,
    orig_h: u32,
}

#[derive(Debug)]
pub struct YoloModel {
    backend: YoloOnnxBackend,
    config: YoloConfig,
    onnx_path: String,
}

impl YoloModel {
    pub fn init_from_spec(spec: &LoadSpec) -> Result<Self> {
        let config = YoloConfig::default();
        match spec.resolved_artifact() {
            ArtifactKind::Onnx => {
                let onnx_path = spec
                    .paths
                    .onnx_path
                    .as_deref()
                    .ok_or_else(|| anyhow!("onnx_path is required for yolo onnx"))?;
                Ok(Self {
                    backend: YoloOnnxBackend::load(onnx_path, &config)?,
                    config,
                    onnx_path: onnx_path.to_string(),
                })
            }
            ArtifactKind::Auto => unreachable!("artifact kind should be resolved before init"),
            other => Err(anyhow!("yolo does not support artifact {:?}", other)),
        }
    }

    pub fn predict(&mut self, source: &str) -> Result<Vec<YoloResults>> {
        self.predict_with_options(source, &YoloPredictOptions::default())
    }

    pub fn predict_with_options(
        &mut self,
        source: &str,
        options: &YoloPredictOptions,
    ) -> Result<Vec<YoloResults>> {
        if options.stream && options.max_frames.is_none() && is_unbounded_stream_source(source) {
            return Err(anyhow!(
                "unbounded stream source requires max_frames when collecting Vec results; use streaming callback API instead"
            ));
        }

        let sources = resolve_image_sources(source, options)?;
        if sources.is_empty() {
            return Err(anyhow!("no supported image sources found for {}", source));
        }

        if options.stream {
            let mut results = Vec::with_capacity(sources.len());
            for item in sources {
                results.push(self.backend.predict(&item, &self.config)?);
            }
            return Ok(results);
        }

        let workers = options.workers.max(1);
        if workers == 1 || sources.len() <= 1 {
            return self.predict_sequential(sources, options.batch_size.max(1));
        }

        self.predict_parallel(sources, workers, options.batch_size.max(1))
    }

    pub fn predict_stream_with_options<F>(
        &mut self,
        source: &str,
        options: &YoloPredictOptions,
        mut on_result: F,
    ) -> Result<()>
    where
        F: FnMut(&YoloResults) -> Result<bool>,
    {
        if options.stream && is_stream_like_source(source) {
            return infer_stream_source(
                source,
                options,
                &mut self.backend,
                &self.config,
                |result| on_result(result),
            );
        }

        let batch_results = self.predict_with_options(source, options)?;
        for result in &batch_results {
            if !on_result(result)? {
                break;
            }
        }
        Ok(())
    }

    pub fn results_to_json(results: &[YoloResults]) -> Result<String> {
        let payload = results
            .iter()
            .map(|result| {
                json!({
                    "boxes": result.boxes,
                    "path": result.path,
                    "names": result.names,
                    "speed": result.speed,
                    "width": result.width,
                    "height": result.height,
                })
            })
            .collect::<Vec<_>>();
        Ok(serde_json::to_string_pretty(&payload)?)
    }

    fn predict_sequential(
        &mut self,
        sources: Vec<ResolvedImageSource>,
        batch_size: usize,
    ) -> Result<Vec<YoloResults>> {
        let mut results = Vec::with_capacity(sources.len());
        for chunk in sources.chunks(batch_size) {
            for item in chunk {
                results.push(self.backend.predict(item, &self.config)?);
            }
        }
        Ok(results)
    }

    fn predict_parallel(
        &self,
        sources: Vec<ResolvedImageSource>,
        workers: usize,
        batch_size: usize,
    ) -> Result<Vec<YoloResults>> {
        let indexed_sources = sources.into_iter().enumerate().collect::<Vec<_>>();
        let chunk_size = batch_size.max(indexed_sources.len().div_ceil(workers).max(1));
        let grouped = indexed_sources
            .chunks(chunk_size)
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>();

        let mut joined = Vec::<(usize, YoloResults)>::new();
        thread::scope(|scope| -> Result<()> {
            let mut handles = Vec::with_capacity(grouped.len());
            for group in grouped {
                let onnx_path = self.onnx_path.clone();
                let config = self.config.clone();
                handles.push(scope.spawn(move || -> Result<Vec<(usize, YoloResults)>> {
                    let mut backend = YoloOnnxBackend::load(&onnx_path, &config)?;
                    let mut partial = Vec::with_capacity(group.len());
                    for (idx, item) in group {
                        partial.push((idx, backend.predict(&item, &config)?));
                    }
                    Ok(partial)
                }));
            }

            for handle in handles {
                let partial = handle
                    .join()
                    .map_err(|_| anyhow!("yolo worker thread panicked"))??;
                joined.extend(partial);
            }
            Ok(())
        })?;

        joined.sort_by_key(|(idx, _)| *idx);
        Ok(joined.into_iter().map(|(_, result)| result).collect())
    }
}

#[derive(Debug)]
struct YoloOnnxBackend {
    #[cfg(feature = "onnx-runtime")]
    session: ort::session::Session,
    #[cfg(not(feature = "onnx-runtime"))]
    _session: (),
    input_name: String,
    output_name: String,
    input_height: usize,
    input_width: usize,
    class_names: Vec<String>,
}

impl YoloOnnxBackend {
    fn load(path: &str, config: &YoloConfig) -> Result<Self> {
        #[cfg(feature = "onnx-runtime")]
        {
            let bundle = create_session(path, None)?;
            let input = bundle
                .session
                .inputs()
                .first()
                .ok_or_else(|| anyhow!("yolo onnx model has no inputs"))?;
            let (input_height, input_width) = match input.dtype() {
                ort::value::ValueType::Tensor { shape, .. } if shape.len() >= 4 => {
                    let h = shape[2].max(1) as usize;
                    let w = shape[3].max(1) as usize;
                    (h, w)
                }
                _ => (config.image_size, config.image_size),
            };
            let output_name = bundle
                .output_names
                .first()
                .cloned()
                .ok_or_else(|| anyhow!("yolo onnx output name is missing"))?;
            let input_name = bundle
                .input_names
                .first()
                .cloned()
                .ok_or_else(|| anyhow!("yolo onnx input name is missing"))?;
            Ok(Self {
                session: bundle.session,
                input_name,
                output_name,
                input_height,
                input_width,
                class_names: config.class_names.clone(),
            })
        }
        #[cfg(not(feature = "onnx-runtime"))]
        {
            let _ = (path, config);
            Err(anyhow!(
                "onnx runtime support is not enabled; rebuild with --features onnx-runtime"
            ))
        }
    }

    fn predict(&mut self, item: &ResolvedImageSource, config: &YoloConfig) -> Result<YoloResults> {
        let preprocess_start = Instant::now();
        let (input_tensor, meta) =
            preprocess_image(&item.image, self.input_width, self.input_height)?;
        let preprocess_ms = preprocess_start.elapsed().as_secs_f32() * 1000.0;

        #[cfg(feature = "onnx-runtime")]
        {
            let inference_start = Instant::now();
            let outputs = self.session.run(vec![(
                self.input_name.clone(),
                ort::value::Tensor::from_array((
                    vec![
                        1_i64,
                        3_i64,
                        self.input_height as i64,
                        self.input_width as i64,
                    ],
                    input_tensor,
                ))?
                .into_dyn(),
            )])?;
            let inference_ms = inference_start.elapsed().as_secs_f32() * 1000.0;
            let output_value = outputs
                .get(&self.output_name)
                .ok_or_else(|| anyhow!("yolo onnx output tensor is missing"))?;
            let postprocess_start = Instant::now();
            let (shape, values) = output_value.try_extract_tensor::<f32>()?;
            let boxes = decode_yolo_output(
                &shape,
                values,
                &meta,
                &self.class_names,
                config.confidence_threshold,
                config.iou_threshold,
                config.max_detections,
            )?;
            let postprocess_ms = postprocess_start.elapsed().as_secs_f32() * 1000.0;
            Ok(YoloResults {
                boxes,
                path: item.source.clone(),
                names: self.class_names.clone(),
                speed: YoloSpeed {
                    preprocess_ms,
                    inference_ms,
                    postprocess_ms,
                },
                width: item.image.width(),
                height: item.image.height(),
                orig_img: Some(item.image.clone()),
            })
        }
        #[cfg(not(feature = "onnx-runtime"))]
        {
            let _ = (item, config, preprocess_ms, input_tensor, meta);
            Err(anyhow!(
                "onnx runtime support is not enabled; rebuild with --features onnx-runtime"
            ))
        }
    }
}

fn preprocess_image(
    image: &DynamicImage,
    input_width: usize,
    input_height: usize,
) -> Result<(Vec<f32>, LetterboxMeta)> {
    let orig_w = image.width();
    let orig_h = image.height();
    if orig_w == 0 || orig_h == 0 {
        return Err(anyhow!("image has invalid size {}x{}", orig_w, orig_h));
    }
    let scale = (input_width as f32 / orig_w as f32).min(input_height as f32 / orig_h as f32);
    let resized_w = ((orig_w as f32 * scale).round() as u32).max(1);
    let resized_h = ((orig_h as f32 * scale).round() as u32).max(1);
    let resized = image
        .resize_exact(resized_w, resized_h, imageops::FilterType::CatmullRom)
        .to_rgba8();
    let mut canvas = RgbaImage::from_pixel(
        input_width as u32,
        input_height as u32,
        Rgba([114, 114, 114, 255]),
    );
    let pad_x = ((input_width as u32).saturating_sub(resized_w)) / 2;
    let pad_y = ((input_height as u32).saturating_sub(resized_h)) / 2;
    imageops::replace(&mut canvas, &resized, pad_x as i64, pad_y as i64);
    let rgb = DynamicImage::ImageRgba8(canvas).to_rgb8();
    let raw = rgb.as_raw();
    let plane = input_width * input_height;
    let mut tensor = vec![0.0_f32; plane * 3];
    for index in 0..plane {
        let src = index * 3;
        tensor[index] = raw[src] as f32 / 255.0;
        tensor[plane + index] = raw[src + 1] as f32 / 255.0;
        tensor[plane * 2 + index] = raw[src + 2] as f32 / 255.0;
    }
    Ok((
        tensor,
        LetterboxMeta {
            scale,
            pad_x: pad_x as f32,
            pad_y: pad_y as f32,
            orig_w,
            orig_h,
        },
    ))
}

fn decode_yolo_output(
    shape: &[i64],
    values: &[f32],
    meta: &LetterboxMeta,
    class_names: &[String],
    conf_threshold: f32,
    iou_threshold: f32,
    max_detections: usize,
) -> Result<Vec<YoloBox>> {
    if shape.len() != 3 || shape[0] != 1 {
        return Err(anyhow!("unsupported yolo output shape: {:?}", shape));
    }
    let dim1 = shape[1].max(1) as usize;
    let dim2 = shape[2].max(1) as usize;
    let (pred_count, attr_count, transposed) = if dim1 > dim2 && dim2 >= 6 {
        (dim2, dim1, true)
    } else {
        (dim1, dim2, false)
    };
    let mut candidates = Vec::new();
    for pred_index in 0..pred_count {
        let prediction = if transposed {
            (0..attr_count)
                .map(|attr_index| values[attr_index * pred_count + pred_index])
                .collect::<Vec<_>>()
        } else {
            let start = pred_index * attr_count;
            values[start..start + attr_count].to_vec()
        };
        if let Some(detection) =
            decode_single_prediction(&prediction, meta, class_names, conf_threshold)
        {
            candidates.push(detection);
        }
    }
    candidates.sort_by(|left, right| right.conf.total_cmp(&left.conf));
    Ok(non_max_suppression(
        candidates,
        iou_threshold,
        max_detections,
    ))
}

fn decode_single_prediction(
    prediction: &[f32],
    meta: &LetterboxMeta,
    class_names: &[String],
    conf_threshold: f32,
) -> Option<YoloBox> {
    if prediction.len() < 6 {
        return None;
    }
    if prediction.len() == 6 {
        let xyxy = if prediction[2] > prediction[0] && prediction[3] > prediction[1] {
            restore_box_xyxy(
                [prediction[0], prediction[1], prediction[2], prediction[3]],
                meta,
            )
        } else {
            restore_box_xywh(
                [prediction[0], prediction[1], prediction[2], prediction[3]],
                meta,
            )
        };
        let conf = prediction[4];
        if conf < conf_threshold {
            return None;
        }
        let cls = prediction[5].max(0.0) as usize;
        let label = class_names
            .get(cls)
            .cloned()
            .unwrap_or_else(|| cls.to_string());
        return Some(YoloBox {
            xywh: xyxy_to_xywh(xyxy),
            xyxy,
            conf,
            cls,
            label,
        });
    }

    let (objectness, class_scores) = if prediction.len() == class_names.len() + 5 {
        (prediction[4], &prediction[5..])
    } else {
        (1.0, &prediction[4..])
    };
    let (cls, class_score) = class_scores
        .iter()
        .copied()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(&right.1))?;
    let conf = class_score * objectness;
    if conf < conf_threshold {
        return None;
    }
    let xyxy = restore_box_xywh(
        [prediction[0], prediction[1], prediction[2], prediction[3]],
        meta,
    );
    Some(YoloBox {
        xywh: xyxy_to_xywh(xyxy),
        xyxy,
        conf,
        cls,
        label: class_names
            .get(cls)
            .cloned()
            .unwrap_or_else(|| cls.to_string()),
    })
}

fn restore_box_xywh(coords: [f32; 4], meta: &LetterboxMeta) -> [f32; 4] {
    let x1 = coords[0] - coords[2] / 2.0;
    let y1 = coords[1] - coords[3] / 2.0;
    let x2 = coords[0] + coords[2] / 2.0;
    let y2 = coords[1] + coords[3] / 2.0;
    restore_box_xyxy([x1, y1, x2, y2], meta)
}

fn restore_box_xyxy(coords: [f32; 4], meta: &LetterboxMeta) -> [f32; 4] {
    [
        ((coords[0] - meta.pad_x) / meta.scale).clamp(0.0, meta.orig_w as f32),
        ((coords[1] - meta.pad_y) / meta.scale).clamp(0.0, meta.orig_h as f32),
        ((coords[2] - meta.pad_x) / meta.scale).clamp(0.0, meta.orig_w as f32),
        ((coords[3] - meta.pad_y) / meta.scale).clamp(0.0, meta.orig_h as f32),
    ]
}

fn xyxy_to_xywh(xyxy: [f32; 4]) -> [f32; 4] {
    [
        (xyxy[0] + xyxy[2]) / 2.0,
        (xyxy[1] + xyxy[3]) / 2.0,
        (xyxy[2] - xyxy[0]).max(0.0),
        (xyxy[3] - xyxy[1]).max(0.0),
    ]
}

fn non_max_suppression(
    mut detections: Vec<YoloBox>,
    iou_threshold: f32,
    max_detections: usize,
) -> Vec<YoloBox> {
    let mut selected = Vec::new();
    while !detections.is_empty() && selected.len() < max_detections {
        let current = detections.remove(0);
        detections.retain(|candidate| {
            candidate.cls != current.cls
                || intersection_over_union(&current.xyxy, &candidate.xyxy) < iou_threshold
        });
        selected.push(current);
    }
    selected
}

fn intersection_over_union(left: &[f32; 4], right: &[f32; 4]) -> f32 {
    let x1 = left[0].max(right[0]);
    let y1 = left[1].max(right[1]);
    let x2 = left[2].min(right[2]);
    let y2 = left[3].min(right[3]);
    let inter_w = (x2 - x1).max(0.0);
    let inter_h = (y2 - y1).max(0.0);
    let inter = inter_w * inter_h;
    let left_area = (left[2] - left[0]).max(0.0) * (left[3] - left[1]).max(0.0);
    let right_area = (right[2] - right[0]).max(0.0) * (right[3] - right[1]).max(0.0);
    let union = left_area + right_area - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

fn draw_box(image: &mut RgbaImage, detection: &YoloBox, color: Rgba<u8>) {
    let width = image.width();
    let height = image.height();
    let x1 = detection.xyxy[0]
        .floor()
        .clamp(0.0, width.saturating_sub(1) as f32) as u32;
    let y1 = detection.xyxy[1]
        .floor()
        .clamp(0.0, height.saturating_sub(1) as f32) as u32;
    let x2 = detection.xyxy[2]
        .ceil()
        .clamp(0.0, width.saturating_sub(1) as f32) as u32;
    let y2 = detection.xyxy[3]
        .ceil()
        .clamp(0.0, height.saturating_sub(1) as f32) as u32;
    for thickness in 0..2 {
        let top = y1.saturating_add(thickness).min(height.saturating_sub(1));
        let bottom = y2.saturating_sub(thickness).min(height.saturating_sub(1));
        let left = x1.saturating_add(thickness).min(width.saturating_sub(1));
        let right = x2.saturating_sub(thickness).min(width.saturating_sub(1));
        for x in left..=right {
            image.put_pixel(x, top, color);
            image.put_pixel(x, bottom, color);
        }
        for y in top..=bottom {
            image.put_pixel(left, y, color);
            image.put_pixel(right, y, color);
        }
    }
    let label = format!("{} {:.2}", detection.label, detection.conf);
    draw_label(image, &label, x1, y1, color);
}

fn draw_label(image: &mut RgbaImage, label: &str, x: u32, y: u32, bg_color: Rgba<u8>) {
    if label.is_empty() || image.width() == 0 || image.height() == 0 {
        return;
    }
    let glyph_w = 8_u32;
    let glyph_h = 8_u32;
    let padding = 2_u32;
    let max_chars = ((image.width().saturating_sub(x)) / glyph_w).max(1) as usize;
    let text = label.chars().take(max_chars).collect::<String>();
    if text.is_empty() {
        return;
    }

    let text_w = text.chars().count() as u32 * glyph_w;
    let box_w = (text_w + padding * 2).min(image.width().saturating_sub(x));
    let box_h = (glyph_h + padding * 2).min(image.height());
    let box_y = y.saturating_sub(box_h.saturating_add(1));

    fill_rect(image, x, box_y, box_w, box_h, bg_color);
    draw_text_8x8(
        image,
        x.saturating_add(padding),
        box_y.saturating_add(padding),
        &text,
        Rgba([255, 255, 255, 255]),
    );
}

fn fill_rect(image: &mut RgbaImage, x: u32, y: u32, w: u32, h: u32, color: Rgba<u8>) {
    let width = image.width();
    let height = image.height();
    if x >= width || y >= height || w == 0 || h == 0 {
        return;
    }
    let x_end = x.saturating_add(w).min(width);
    let y_end = y.saturating_add(h).min(height);
    for yy in y..y_end {
        for xx in x..x_end {
            image.put_pixel(xx, yy, color);
        }
    }
}

fn draw_text_8x8(image: &mut RgbaImage, x: u32, y: u32, text: &str, color: Rgba<u8>) {
    let mut cursor_x = x;
    let width = image.width();
    let height = image.height();
    for ch in text.chars() {
        let c = if ch.is_ascii() { ch } else { '?' };
        if let Some(glyph) = BASIC_FONTS.get(c) {
            for (row, row_bits) in glyph.iter().enumerate() {
                let yy = y.saturating_add(row as u32);
                if yy >= height {
                    continue;
                }
                for col in 0..8 {
                    if ((row_bits >> col) & 1) == 0 {
                        continue;
                    }
                    let xx = cursor_x.saturating_add(col as u32);
                    if xx < width {
                        image.put_pixel(xx, yy, color);
                    }
                }
            }
        }
        cursor_x = cursor_x.saturating_add(8);
        if cursor_x >= width {
            break;
        }
    }
}

fn resolve_image_sources(
    source: &str,
    options: &YoloPredictOptions,
) -> Result<Vec<ResolvedImageSource>> {
    if source.eq_ignore_ascii_case("screen") {
        return Err(anyhow!(
            "screen capture is not supported by the native onnx backend yet"
        ));
    }
    if source.contains("youtube.com") || source.contains("youtu.be") {
        return Err(anyhow!(
            "youtube inputs are not supported by the native onnx backend yet"
        ));
    }
    if parse_webcam_index(source).is_some() {
        if !options.stream {
            return Err(anyhow!("webcam numeric index input requires stream mode"));
        }
        return resolve_stream_sources(source, options);
    }

    if is_network_stream_source(source) {
        if !options.stream {
            return Err(anyhow!(
                "real-time stream input requires stream mode; retry with --stream"
            ));
        }
        return resolve_stream_sources(source, options);
    }

    if source.starts_with("http://") || source.starts_with("https://") {
        if is_supported_image_path(source) {
            return Ok(vec![ResolvedImageSource {
                source: source.to_string(),
                image: load_image_from_url(source)?,
            }]);
        }
        if options.stream {
            return resolve_stream_sources(source, options);
        }
        return Err(anyhow!(
            "only image urls are supported in non-stream mode; use --stream for video urls"
        ));
    }

    if source.starts_with("file://") {
        let path = get_file_path(source)?;
        return resolve_local_path(&path, options);
    }

    if contains_glob_pattern(source) {
        let mut sources = Vec::new();
        for entry in glob(source)? {
            let path = entry?;
            sources.extend(resolve_local_path(&path, options)?);
        }
        return Ok(sources);
    }

    resolve_local_path(Path::new(source), options)
}

fn resolve_local_path(
    path: &Path,
    options: &YoloPredictOptions,
) -> Result<Vec<ResolvedImageSource>> {
    if !path.exists() {
        return Err(anyhow!("source path not found: {}", path.display()));
    }
    if path.is_dir() {
        let mut stack = vec![path.to_path_buf()];
        let mut resolved = Vec::new();
        while let Some(current) = stack.pop() {
            for entry in std::fs::read_dir(&current)? {
                let entry = entry?;
                let child = entry.path();
                if child.is_dir() {
                    stack.push(child);
                } else if is_supported_image_path(&child) {
                    resolved.push(load_image_source_from_path(&child)?);
                } else if is_supported_video_path(&child) {
                    if !options.stream {
                        return Err(anyhow!(
                            "video sources require stream mode; retry with --stream: {}",
                            child.display()
                        ));
                    }
                    resolved.extend(resolve_stream_sources(&child.to_string_lossy(), options)?);
                }
            }
        }
        resolved.sort_by(|left, right| left.source.cmp(&right.source));
        return Ok(resolved);
    }

    if is_supported_list_file(path) {
        let mut resolved = Vec::new();
        let content = std::fs::read_to_string(path)?;
        for line in content.lines() {
            let entry = normalize_list_entry(line);
            if entry.is_empty() {
                continue;
            }
            resolved.extend(resolve_image_sources(&entry, options)?);
        }
        return Ok(resolved);
    }

    if is_supported_image_path(path) {
        return Ok(vec![load_image_source_from_path(path)?]);
    }
    if is_supported_video_path(path) {
        if !options.stream {
            return Err(anyhow!(
                "video sources require stream mode; retry with --stream: {}",
                path.display()
            ));
        }
        return resolve_stream_sources(&path.to_string_lossy(), options);
    }
    Err(anyhow!("unsupported source path: {}", path.display()))
}

#[cfg(feature = "ffmpeg")]
fn resolve_stream_sources(
    source: &str,
    options: &YoloPredictOptions,
) -> Result<Vec<ResolvedImageSource>> {
    let mut results = Vec::new();
    if options.max_frames.is_none() && is_unbounded_stream_source(source) {
        return Err(anyhow!(
            "collecting unbounded stream frames requires max_frames; use streaming callback API instead"
        ));
    }
    decode_stream_frames(source, options, |item| {
        results.push(item);
        Ok(true)
    })?;

    if results.is_empty() {
        return Err(anyhow!(
            "no video frames extracted from stream source {source}; check url/codec availability"
        ));
    }
    Ok(results)
}

#[cfg(not(feature = "ffmpeg"))]
fn resolve_stream_sources(
    _source: &str,
    _options: &YoloPredictOptions,
) -> Result<Vec<ResolvedImageSource>> {
    Err(anyhow!(
        "stream/video/rtsp input requires ffmpeg support; rebuild with --features ffmpeg"
    ))
}

#[cfg(feature = "ffmpeg")]
fn infer_stream_source<F>(
    source: &str,
    options: &YoloPredictOptions,
    backend: &mut YoloOnnxBackend,
    config: &YoloConfig,
    mut on_result: F,
) -> Result<()>
where
    F: FnMut(&YoloResults) -> Result<bool>,
{
    let mut emitted = 0_usize;
    decode_stream_frames(source, options, |item| {
        let result = backend.predict(&item, config)?;
        emitted += 1;
        on_result(&result)
    })?;
    if emitted == 0 {
        return Err(anyhow!(
            "no stream frames produced prediction results for source {source}"
        ));
    }
    Ok(())
}

#[cfg(not(feature = "ffmpeg"))]
fn infer_stream_source<F>(
    _source: &str,
    _options: &YoloPredictOptions,
    _backend: &mut YoloOnnxBackend,
    _config: &YoloConfig,
    _on_result: F,
) -> Result<()>
where
    F: FnMut(&YoloResults) -> Result<bool>,
{
    Err(anyhow!(
        "stream/video/rtsp input requires ffmpeg support; rebuild with --features ffmpeg"
    ))
}

#[cfg(feature = "ffmpeg")]
fn decode_stream_frames<F>(
    source: &str,
    options: &YoloPredictOptions,
    mut on_frame: F,
) -> Result<()>
where
    F: FnMut(ResolvedImageSource) -> Result<bool>,
{
    let mut ictx = open_stream_input_context(source, options)?;
    let input_stream = ictx
        .streams()
        .best(ffmpeg::media::Type::Video)
        .ok_or_else(|| anyhow!("no video stream found for source {source}"))?;
    let video_stream_index = input_stream.index();

    let context_decoder =
        ffmpeg::codec::context::Context::from_parameters(input_stream.parameters())
            .map_err(|e| anyhow!("failed to create decoder context for {source}: {e}"))?;
    let mut decoder = context_decoder
        .decoder()
        .video()
        .map_err(|e| anyhow!("failed to create video decoder for {source}: {e}"))?;

    let src_w = decoder.width();
    let src_h = decoder.height();
    if src_w == 0 || src_h == 0 {
        return Err(anyhow!("stream source has invalid dimensions: {source}"));
    }

    let mut scaler = ffmpeg::software::scaling::context::Context::get(
        decoder.format(),
        src_w,
        src_h,
        ffmpeg::format::Pixel::RGB24,
        src_w,
        src_h,
        ffmpeg::software::scaling::flag::Flags::BILINEAR
            | ffmpeg::software::scaling::flag::Flags::ACCURATE_RND,
    )
    .map_err(|e| anyhow!("failed to create frame scaler for {source}: {e}"))?;

    let stride = options.frame_stride.max(1);
    let max_frames = options.max_frames;
    let mut frame_index = 0_usize;
    let mut kept_frames = 0_usize;
    let mut stop = false;

    let mut receive_and_process = |decoder: &mut ffmpeg::decoder::Video| -> Result<bool> {
        let mut decoded = ffmpeg::frame::Video::empty();
        while decoder.receive_frame(&mut decoded).is_ok() {
            if stop_requested(options) {
                return Ok(false);
            }
            if frame_index.is_multiple_of(stride) {
                let mut rgb_frame = ffmpeg::frame::Video::empty();
                scaler
                    .run(&decoded, &mut rgb_frame)
                    .map_err(|e| anyhow!("failed to scale video frame for {source}: {e}"))?;
                let image = rgb_frame_to_dynamic_image(&rgb_frame)?;
                let keep_going = on_frame(ResolvedImageSource {
                    source: format!("{source}#frame={frame_index}"),
                    image,
                })?;
                kept_frames += 1;
                if !keep_going || max_frames.is_some_and(|limit| kept_frames >= limit) {
                    return Ok(false);
                }
            }
            frame_index += 1;
        }
        Ok(true)
    };

    for (stream, packet) in ictx.packets() {
        if stop_requested(options) || stop {
            break;
        }
        if stream.index() != video_stream_index {
            continue;
        }
        decoder
            .send_packet(&packet)
            .map_err(|e| anyhow!("failed to feed packet into decoder for {source}: {e}"))?;
        if !receive_and_process(&mut decoder)? {
            stop = true;
            break;
        }
    }

    if !stop && !stop_requested(options) {
        decoder
            .send_eof()
            .map_err(|e| anyhow!("failed to flush decoder for {source}: {e}"))?;
        let _ = receive_and_process(&mut decoder)?;
    }

    Ok(())
}

#[cfg(feature = "ffmpeg")]
fn open_stream_input_context(
    source: &str,
    options: &YoloPredictOptions,
) -> Result<ffmpeg::format::context::Input> {
    ffmpeg_init_once()?;

    if let Some(webcam_index) = parse_webcam_index(source) {
        let (capture_source, capture_options) = webcam_capture_source(webcam_index);
        return ffmpeg::format::input_with_dictionary(&capture_source, capture_options)
            .map_err(|e| anyhow!("failed to open webcam source {capture_source}: {e}"));
    }

    let source_owned = source.to_string();
    let stop_flag = options.stop_flag.clone();
    ffmpeg::format::input_with_interrupt(&source_owned, move || {
        stop_flag
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::Relaxed))
    })
    .map_err(|e| anyhow!("failed to open stream source {source_owned}: {e}"))
}

#[cfg(feature = "ffmpeg")]
fn webcam_capture_source(index: usize) -> (String, ffmpeg::Dictionary<'static>) {
    let mut options = ffmpeg::Dictionary::new();
    let env_specific = format!("AHA_YOLO_WEBCAM_SOURCE_{index}");
    if let Ok(source) = std::env::var(&env_specific) {
        if let Ok(format) = std::env::var("AHA_YOLO_WEBCAM_FORMAT") {
            options.set("f", &format);
        }
        return (source, options);
    }
    if let Ok(source) = std::env::var("AHA_YOLO_WEBCAM_SOURCE") {
        if let Ok(format) = std::env::var("AHA_YOLO_WEBCAM_FORMAT") {
            options.set("f", &format);
        }
        return (source, options);
    }
    #[cfg(target_os = "windows")]
    {
        options.set("f", "dshow");
        (format!("video={index}"), options)
    }
    #[cfg(target_os = "linux")]
    {
        options.set("f", "v4l2");
        (format!("/dev/video{index}"), options)
    }
    #[cfg(target_os = "macos")]
    {
        options.set("f", "avfoundation");
        (format!("{index}:none"), options)
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
    {
        (index.to_string(), options)
    }
}

#[cfg(feature = "ffmpeg")]
fn stop_requested(options: &YoloPredictOptions) -> bool {
    options
        .stop_flag
        .as_ref()
        .is_some_and(|flag| flag.load(Ordering::Relaxed))
}

#[cfg(feature = "ffmpeg")]
fn ffmpeg_init_once() -> Result<()> {
    static INIT_RESULT: OnceLock<std::result::Result<(), String>> = OnceLock::new();
    let result = INIT_RESULT.get_or_init(|| ffmpeg::init().map_err(|err| err.to_string()));
    match result {
        Ok(()) => Ok(()),
        Err(error) => Err(anyhow!("failed to initialize ffmpeg runtime: {}", error)),
    }
}

#[cfg(feature = "ffmpeg")]
fn rgb_frame_to_dynamic_image(frame: &ffmpeg::frame::Video) -> Result<DynamicImage> {
    let width = frame.width() as usize;
    let height = frame.height() as usize;
    let src = frame.data(0);
    let stride = frame.stride(0);
    let row_bytes = width * 3;
    let mut packed = vec![0_u8; row_bytes * height];
    for row in 0..height {
        let src_offset = row * stride;
        let dst_offset = row * row_bytes;
        packed[dst_offset..dst_offset + row_bytes]
            .copy_from_slice(&src[src_offset..src_offset + row_bytes]);
    }
    let image = image::RgbImage::from_raw(width as u32, height as u32, packed)
        .ok_or_else(|| anyhow!("failed to convert ffmpeg RGB frame to image"))?;
    Ok(DynamicImage::ImageRgb8(image))
}

fn is_network_stream_source(source: &str) -> bool {
    if source.starts_with("rtsp://")
        || source.starts_with("rtmp://")
        || source.starts_with("tcp://")
    {
        return true;
    }
    false
}

fn parse_webcam_index(source: &str) -> Option<usize> {
    let trimmed = source.trim();
    if trimmed.is_empty() {
        return None;
    }
    trimmed.parse::<usize>().ok()
}

fn is_unbounded_stream_source(source: &str) -> bool {
    is_network_stream_source(source) || parse_webcam_index(source).is_some()
}

fn is_stream_like_source(source: &str) -> bool {
    if is_unbounded_stream_source(source) {
        return true;
    }
    if source.starts_with("http://") || source.starts_with("https://") {
        return !is_supported_image_path(source);
    }
    if source.starts_with("file://") {
        if let Ok(path) = get_file_path(source) {
            return is_supported_video_path(path);
        }
        return false;
    }
    is_supported_video_path(source)
}

fn normalize_list_entry(line: &str) -> String {
    line.split(',')
        .next()
        .unwrap_or_default()
        .trim()
        .trim_matches('"')
        .to_string()
}

fn load_image_source_from_path(path: &Path) -> Result<ResolvedImageSource> {
    Ok(ResolvedImageSource {
        source: path.to_string_lossy().to_string(),
        image: image::open(path)?,
    })
}

fn contains_glob_pattern(source: &str) -> bool {
    source.contains('*') || source.contains('?') || source.contains('[')
}

fn is_supported_list_file(path: &Path) -> bool {
    path.extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("txt") || ext.eq_ignore_ascii_case("csv"))
}

fn is_supported_image_path(path: impl AsRef<Path>) -> bool {
    path.as_ref().extension().is_some_and(|ext| {
        SUPPORTED_IMAGE_EXTENSIONS
            .iter()
            .any(|candidate| ext.eq_ignore_ascii_case(candidate))
    })
}

fn is_supported_video_path(path: impl AsRef<Path>) -> bool {
    path.as_ref().extension().is_some_and(|ext| {
        SUPPORTED_VIDEO_EXTENSIONS
            .iter()
            .any(|candidate| ext.eq_ignore_ascii_case(candidate))
    })
}
