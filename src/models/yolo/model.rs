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

use super::config::{YoloConfig, YoloTaskKind};

const SUPPORTED_IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"];
const SUPPORTED_VIDEO_EXTENSIONS: &[&str] = &["mp4", "avi", "mov", "mkv", "webm"];

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct YoloKeypoint {
    pub x: f32,
    pub y: f32,
    pub conf: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct YoloMask {
    pub width: u32,
    pub height: u32,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct YoloClassification {
    pub cls: usize,
    pub label: String,
    pub conf: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct YoloObb {
    pub xywhr: [f32; 5],
    pub corners: [[f32; 2]; 4],
    pub xyxy: [f32; 4],
    pub conf: f32,
    pub cls: usize,
    pub label: String,
}

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
    pub task: YoloTaskKind,
    pub boxes: Vec<YoloBox>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub masks: Vec<YoloMask>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub keypoints: Vec<Vec<YoloKeypoint>>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub probs: Vec<YoloClassification>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub obb: Vec<YoloObb>,
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
        for (index, mask) in self.masks.iter().enumerate() {
            overlay_mask(&mut canvas, mask, palette_color(index), 96);
            draw_mask_contour(&mut canvas, mask, palette_color(index));
        }
        for detection in &self.boxes {
            draw_box(&mut canvas, detection, Rgba([255, 64, 64, 255]));
        }
        for obb in &self.obb {
            draw_oriented_box(&mut canvas, obb, Rgba([64, 200, 255, 255]));
        }
        for points in &self.keypoints {
            draw_keypoints(&mut canvas, points, Rgba([80, 255, 96, 255]));
        }
        if let Some(best) = self.probs.first() {
            draw_label(
                &mut canvas,
                &format!("{} {:.2}", best.label, best.conf),
                4,
                20,
                Rgba([96, 96, 255, 255]),
            );
        }
        Ok(canvas)
    }

    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    pub fn to_coco_annotations(
        &self,
        image_id: u64,
        next_annotation_id: &mut u64,
    ) -> Vec<serde_json::Value> {
        let mut annotations = Vec::new();
        for (index, detection) in self.boxes.iter().enumerate() {
            let mut annotation = json!({
                "id": *next_annotation_id,
                "image_id": image_id,
                "category_id": detection.cls + 1,
                "bbox": [
                    detection.xyxy[0],
                    detection.xyxy[1],
                    detection.xywh[2],
                    detection.xywh[3],
                ],
                "score": detection.conf,
                "area": detection.xywh[2] * detection.xywh[3],
                "iscrowd": 0,
            });
            if let Some(points) = self.keypoints.get(index)
                && !points.is_empty()
            {
                let flat = points
                    .iter()
                    .flat_map(|point| [point.x, point.y, if point.conf > 0.0 { 2.0 } else { 0.0 }])
                    .collect::<Vec<_>>();
                annotation["keypoints"] = json!(flat);
                annotation["num_keypoints"] = json!(points.iter().filter(|point| point.conf > 0.0).count());
            }
            if let Some(mask) = self.masks.get(index) {
                let contour = sample_mask_contour_points(mask, 2, 256)
                    .into_iter()
                    .flat_map(|[x, y]| [x as f32, y as f32])
                    .collect::<Vec<_>>();
                if contour.len() >= 6 {
                    annotation["segmentation"] = json!([contour]);
                }
            }
            annotations.push(annotation);
            *next_annotation_id += 1;
        }
        for obb in &self.obb {
            annotations.push(json!({
                "id": *next_annotation_id,
                "image_id": image_id,
                "category_id": obb.cls + 1,
                "bbox": [
                    obb.xyxy[0],
                    obb.xyxy[1],
                    (obb.xyxy[2] - obb.xyxy[0]).max(0.0),
                    (obb.xyxy[3] - obb.xyxy[1]).max(0.0),
                ],
                "score": obb.conf,
                "area": (obb.xyxy[2] - obb.xyxy[0]).max(0.0) * (obb.xyxy[3] - obb.xyxy[1]).max(0.0),
                "iscrowd": 0,
                "segmentation": [obb.corners.iter().flat_map(|point| [point[0], point[1]]).collect::<Vec<_>>()],
            }));
            *next_annotation_id += 1;
        }
        annotations
    }

    pub fn latency_ms(&self) -> f32 {
        self.speed.preprocess_ms + self.speed.inference_ms + self.speed.postprocess_ms
    }

    pub fn save_txt(&self, path: &str) -> Result<()> {
        let mut lines = Vec::new();
        // Add task-type header for disambiguation
        lines.push(format!("# task: {:?}", self.task));
        if !self.probs.is_empty() {
            for prediction in &self.probs {
                lines.push(format!(
                    "{} {:.6} {}",
                    prediction.cls, prediction.conf, prediction.label
                ));
            }
        } else if !self.obb.is_empty() {
            for obb in &self.obb {
                lines.push(format!(
                    "{} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}",
                    obb.cls,
                    obb.xywhr[0],
                    obb.xywhr[1],
                    obb.xywhr[2],
                    obb.xywhr[3],
                    obb.xywhr[4],
                    obb.conf,
                ));
            }
        } else {
            for (index, detection) in self.boxes.iter().enumerate() {
                let mut fields = vec![
                    detection.cls.to_string(),
                    format!("{:.6}", detection.xywh[0]),
                    format!("{:.6}", detection.xywh[1]),
                    format!("{:.6}", detection.xywh[2]),
                    format!("{:.6}", detection.xywh[3]),
                    format!("{:.6}", detection.conf),
                ];
                if let Some(points) = self.keypoints.get(index)
                    && !points.is_empty()
                {
                    fields.push("kpts".to_string());
                    for point in points {
                        fields.push(format!("{:.6}", point.x));
                        fields.push(format!("{:.6}", point.y));
                        fields.push(format!("{:.6}", point.conf));
                    }
                }
                if let Some(mask) = self.masks.get(index) {
                    let contour = sample_mask_contour_points(mask, 4, 128);
                    if !contour.is_empty() {
                        fields.push("mask".to_string());
                        for [x, y] in contour {
                            fields.push(x.to_string());
                            fields.push(y.to_string());
                        }
                    }
                }
                lines.push(fields.join(" "));
            }
        }
        std::fs::write(path, lines.join("\n"))?;
        Ok(())
    }

    /// Discard the original image to free memory. Useful after batch processing
    /// when only the structured results (boxes, masks, etc.) are needed.
    pub fn strip_images(&mut self) {
        self.orig_img = None;
    }
}

#[derive(Debug, Clone)]
pub struct YoloPredictOptions {
    pub stream: bool,
    pub workers: usize,
    pub batch_size: usize,
    pub top_k: Option<usize>,
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
            top_k: None,
            max_frames: None,
            frame_stride: 1,
            stop_flag: None,
        }
    }
}

fn palette_color(index: usize) -> Rgba<u8> {
    const COLORS: [[u8; 4]; 6] = [
        [255, 99, 132, 255],
        [54, 162, 235, 255],
        [255, 206, 86, 255],
        [75, 192, 192, 255],
        [153, 102, 255, 255],
        [255, 159, 64, 255],
    ];
    Rgba(COLORS[index % COLORS.len()])
}

fn overlay_mask(image: &mut RgbaImage, mask: &YoloMask, color: Rgba<u8>, alpha: u8) {
    if mask.width != image.width() || mask.height != image.height() {
        return;
    }
    let alpha_factor = alpha as f32 / 255.0;
    for (index, pixel) in image.pixels_mut().enumerate() {
        let Some(&mask_value) = mask.data.get(index) else {
            break;
        };
        if mask_value == 0 {
            continue;
        }
        let weight = (mask_value as f32 / 255.0) * alpha_factor;
        for channel in 0..3 {
            pixel.0[channel] = ((pixel.0[channel] as f32 * (1.0 - weight))
                + (color.0[channel] as f32 * weight))
                .round()
                .clamp(0.0, 255.0) as u8;
        }
    }
}

fn draw_mask_contour(image: &mut RgbaImage, mask: &YoloMask, color: Rgba<u8>) {
    for [x, y] in sample_mask_contour_points(mask, 1, usize::MAX) {
        if x < image.width() && y < image.height() {
            image.put_pixel(x, y, color);
        }
    }
}

fn draw_line(image: &mut RgbaImage, start: [f32; 2], end: [f32; 2], color: Rgba<u8>) {
    let mut x0 = start[0].round() as i32;
    let mut y0 = start[1].round() as i32;
    let x1 = end[0].round() as i32;
    let y1 = end[1].round() as i32;
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    loop {
        if x0 >= 0 && y0 >= 0 && (x0 as u32) < image.width() && (y0 as u32) < image.height() {
            image.put_pixel(x0 as u32, y0 as u32, color);
        }
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = err * 2;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

fn draw_oriented_box(image: &mut RgbaImage, obb: &YoloObb, color: Rgba<u8>) {
    for index in 0..4 {
        draw_line(image, obb.corners[index], obb.corners[(index + 1) % 4], color);
    }
    let anchor_x = obb.corners[0][0].max(0.0) as u32;
    let anchor_y = obb.corners[0][1].max(0.0) as u32;
    draw_label(image, &format!("{} {:.2}", obb.label, obb.conf), anchor_x, anchor_y, color);
}

fn draw_keypoints(image: &mut RgbaImage, keypoints: &[YoloKeypoint], color: Rgba<u8>) {
    for &(left, right) in coco_pose_skeleton(keypoints.len()) {
        let Some(start) = keypoints.get(left) else {
            continue;
        };
        let Some(end) = keypoints.get(right) else {
            continue;
        };
        if start.conf > 0.0 && end.conf > 0.0 {
            draw_line(image, [start.x, start.y], [end.x, end.y], color);
        }
    }
    for point in keypoints {
        if point.conf <= 0.0 {
            continue;
        }
        let cx = point.x.round() as i32;
        let cy = point.y.round() as i32;
        for dy in -2..=2 {
            for dx in -2..=2 {
                let x = cx + dx;
                let y = cy + dy;
                if x >= 0 && y >= 0 && (x as u32) < image.width() && (y as u32) < image.height() {
                    image.put_pixel(x as u32, y as u32, color);
                }
            }
        }
    }
}

fn coco_pose_skeleton(keypoint_count: usize) -> &'static [(usize, usize)] {
    const COCO17: &[(usize, usize)] = &[
        (15, 13),
        (13, 11),
        (16, 14),
        (14, 12),
        (11, 12),
        (5, 11),
        (6, 12),
        (5, 6),
        (5, 7),
        (6, 8),
        (7, 9),
        (8, 10),
        (1, 2),
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (3, 5),
        (4, 6),
    ];
    if keypoint_count >= 17 { COCO17 } else { &[] }
}

fn is_mask_boundary(mask: &YoloMask, x: u32, y: u32) -> bool {
    if x >= mask.width || y >= mask.height {
        return false;
    }
    let index = (y * mask.width + x) as usize;
    if mask.data.get(index).copied().unwrap_or(0) == 0 {
        return false;
    }
    for (dx, dy) in [(-1_i32, 0_i32), (1, 0), (0, -1), (0, 1)] {
        let nx = x as i32 + dx;
        let ny = y as i32 + dy;
        if nx < 0 || ny < 0 || nx >= mask.width as i32 || ny >= mask.height as i32 {
            return true;
        }
        let neighbor_index = (ny as u32 * mask.width + nx as u32) as usize;
        if mask.data.get(neighbor_index).copied().unwrap_or(0) == 0 {
            return true;
        }
    }
    false
}

fn sample_mask_contour_points(mask: &YoloMask, stride: usize, max_points: usize) -> Vec<[u32; 2]> {
    if mask.width == 0 || mask.height == 0 {
        return Vec::new();
    }
    let stride = stride.max(1);
    let mut points = Vec::new();
    let mut boundary_index = 0_usize;
    for y in 0..mask.height {
        for x in 0..mask.width {
            if !is_mask_boundary(mask, x, y) {
                continue;
            }
            if boundary_index.is_multiple_of(stride) {
                points.push([x, y]);
                if points.len() >= max_points {
                    return points;
                }
            }
            boundary_index += 1;
        }
    }
    points
}

fn collect_coco_categories(results: &[YoloResults]) -> Vec<serde_json::Value> {
    let mut categories = Vec::<(usize, String)>::new();
    for result in results {
        for (index, name) in result.names.iter().enumerate() {
            if !categories.iter().any(|(cls, _)| *cls == index) {
                categories.push((index, name.clone()));
            }
        }
        for detection in &result.boxes {
            if !categories.iter().any(|(cls, _)| *cls == detection.cls) {
                categories.push((detection.cls, detection.label.clone()));
            }
        }
        for obb in &result.obb {
            if !categories.iter().any(|(cls, _)| *cls == obb.cls) {
                categories.push((obb.cls, obb.label.clone()));
            }
        }
        for prediction in &result.probs {
            if !categories.iter().any(|(cls, _)| *cls == prediction.cls) {
                categories.push((prediction.cls, prediction.label.clone()));
            }
        }
    }
    categories.sort_by_key(|(cls, _)| *cls);
    categories
        .into_iter()
        .map(|(cls, name)| json!({ "id": cls + 1, "name": name }))
        .collect()
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

#[derive(Debug, Clone)]
struct PredictionLayout {
    pred_count: usize,
    attr_count: usize,
    transposed: bool,
}

#[derive(Debug, Clone, Copy)]
struct PredictionAuxLayout {
    has_objectness: bool,
    class_count: usize,
    extra_start: usize,
}

#[derive(Debug, Default)]
struct DecodedYoloOutput {
    task: YoloTaskKind,
    boxes: Vec<YoloBox>,
    masks: Vec<YoloMask>,
    keypoints: Vec<Vec<YoloKeypoint>>,
    probs: Vec<YoloClassification>,
    obb: Vec<YoloObb>,
}

#[derive(Debug, Clone)]
struct OnnxOutputTensor<'a> {
    name: String,
    shape: Vec<i64>,
    values: &'a [f32],
}

#[derive(Debug)]
pub struct YoloModel {
    backend: YoloOnnxBackend,
    config: YoloConfig,
    onnx_path: String,
}

impl YoloModel {
    pub fn init_from_spec(spec: &LoadSpec) -> Result<Self> {
        Self::init_with_config(spec, YoloConfig::default())
    }

    pub fn init_with_config(spec: &LoadSpec, config: YoloConfig) -> Result<Self> {
        match spec.resolved_artifact() {
            ArtifactKind::Onnx => {
                let onnx_path = spec
                    .paths
                    .onnx_path
                    .as_deref()
                    .ok_or_else(|| anyhow!("onnx_path is required for yolo onnx"))?;
                let mut model = Self {
                    backend: YoloOnnxBackend::load(onnx_path, &config)?,
                    config,
                    onnx_path: onnx_path.to_string(),
                };
                model.warmup()?;
                Ok(model)
            }
            ArtifactKind::Auto => unreachable!("artifact kind should be resolved before init"),
            other => Err(anyhow!("yolo does not support artifact {:?}", other)),
        }
    }

    /// Run a single dummy inference to trigger ONNX session lazy initialization,
    /// so that the first real prediction does not include the cold-start overhead.
    fn warmup(&mut self) -> Result<()> {
        let dummy_image = DynamicImage::ImageRgba8(RgbaImage::from_pixel(
            self.config.image_size as u32,
            self.config.image_size as u32,
            Rgba([114, 114, 114, 255]),
        ));
        let source = ResolvedImageSource {
            source: "__warmup__".to_string(),
            image: dummy_image,
        };
        let _ = self.backend.predict(&source, &self.config, None)?;
        Ok(())
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

        let mut results = if options.stream {
            let mut results = Vec::with_capacity(sources.len());
            for item in sources {
                results.push(self.backend.predict(&item, &self.config, options.top_k)?);
            }
            results
        } else {
            let workers = options.workers.max(1);
            if workers == 1 || sources.len() <= 1 {
                self.predict_sequential(sources, options.batch_size.max(1), options.top_k)?
            } else {
                self.predict_parallel(sources, workers, options.batch_size.max(1), options.top_k)?
            }
        };

        if !self.config.keep_images {
            for result in &mut results {
                result.strip_images();
            }
        }

        Ok(results)
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
        Ok(serde_json::to_string_pretty(results)?)
    }

    pub fn results_to_coco_json(results: &[YoloResults]) -> Result<String> {
        let images = results
            .iter()
            .enumerate()
            .map(|(index, result)| {
                json!({
                    "id": index + 1,
                    "file_name": result.path,
                    "width": result.width,
                    "height": result.height,
                })
            })
            .collect::<Vec<_>>();
        let categories = collect_coco_categories(results);
        let mut next_annotation_id = 1_u64;
        let mut annotations = Vec::new();
        for (index, result) in results.iter().enumerate() {
            annotations.extend(result.to_coco_annotations((index + 1) as u64, &mut next_annotation_id));
        }
        Ok(serde_json::to_string_pretty(&json!({
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }))?)
    }

    fn predict_sequential(
        &mut self,
        sources: Vec<ResolvedImageSource>,
        batch_size: usize,
        top_k: Option<usize>,
    ) -> Result<Vec<YoloResults>> {
        // NOTE: Each image is inferred individually with batch_size=1 to the ONNX
        // session because the model input shape is fixed at [1,3,H,W].
        // The batch_size parameter controls chunking for progress reporting only.
        let _ = batch_size;
        let mut results = Vec::with_capacity(sources.len());
        for item in sources {
            results.push(self.backend.predict(&item, &self.config, top_k)?);
        }
        Ok(results)
    }

    fn predict_parallel(
        &self,
        sources: Vec<ResolvedImageSource>,
        workers: usize,
        batch_size: usize,
        top_k: Option<usize>,
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
                let top_k = top_k;
                handles.push(scope.spawn(move || -> Result<Vec<(usize, YoloResults)>> {
                    let mut backend = YoloOnnxBackend::load(&onnx_path, &config)?;
                    let mut partial = Vec::with_capacity(group.len());
                    for (idx, item) in group {
                        partial.push((idx, backend.predict(&item, &config, top_k)?));
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
    output_names: Vec<String>,
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
            if bundle.output_names.is_empty() {
                return Err(anyhow!("yolo onnx output names are missing"));
            }
            let input_name = bundle
                .input_names
                .first()
                .cloned()
                .ok_or_else(|| anyhow!("yolo onnx input name is missing"))?;
            Ok(Self {
                session: bundle.session,
                input_name,
                output_names: bundle.output_names,
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

    fn predict(
        &mut self,
        item: &ResolvedImageSource,
        config: &YoloConfig,
        top_k: Option<usize>,
    ) -> Result<YoloResults> {
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
            let postprocess_start = Instant::now();
            let mut output_tensors = Vec::with_capacity(self.output_names.len());
            for name in &self.output_names {
                let output_value = outputs
                    .get(name)
                    .ok_or_else(|| anyhow!("yolo onnx output tensor is missing: {name}"))?;
                let (shape, values) = output_value.try_extract_tensor::<f32>()?;
                output_tensors.push(OnnxOutputTensor {
                    name: name.clone(),
                    shape: shape.to_vec(),
                    values,
                });
            }
            let decoded = decode_yolo_outputs(
                &output_tensors,
                &meta,
                &self.class_names,
                config.confidence_threshold,
                config.iou_threshold,
                config.max_detections,
                top_k,
                config.task_kind,
                config.nms_class_agnostic,
                config.keypoint_confidence_threshold,
            )?;
            let postprocess_ms = postprocess_start.elapsed().as_secs_f32() * 1000.0;
            Ok(YoloResults {
                task: decoded.task,
                boxes: decoded.boxes,
                masks: decoded.masks,
                keypoints: decoded.keypoints,
                probs: decoded.probs,
                obb: decoded.obb,
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
            let _ = (item, config, top_k, preprocess_ms, input_tensor, meta);
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

fn decode_yolo_outputs(
    outputs: &[OnnxOutputTensor<'_>],
    meta: &LetterboxMeta,
    class_names: &[String],
    conf_threshold: f32,
    iou_threshold: f32,
    max_detections: usize,
    top_k: Option<usize>,
    task_kind_override: Option<YoloTaskKind>,
    nms_class_agnostic: bool,
    keypoint_confidence_threshold: f32,
) -> Result<DecodedYoloOutput> {
    let primary = select_primary_output(outputs)
        .ok_or_else(|| anyhow!("yolo onnx output tensor list is empty"))?;

    // If the user specified a task kind, use it directly (skip auto-detection).
    if let Some(task) = task_kind_override {
        return decode_by_task_kind(
            task, outputs, primary, meta, class_names, conf_threshold,
            iou_threshold, max_detections, top_k, nms_class_agnostic,
            keypoint_confidence_threshold,
        );
    }

    // Auto-detect task kind from output tensors.
    if is_classification_output(primary) {
        return decode_yolo_classify_output(
            primary,
            class_names,
            conf_threshold,
            top_k,
            max_detections,
        );
    }

    if let Some(proto) = find_mask_proto_output(outputs) {
        if let Some(decoded) = try_decode_yolo_segment_output(
            primary,
            proto,
            meta,
            class_names,
            conf_threshold,
            iou_threshold,
            max_detections,
            nms_class_agnostic,
        )? {
            return Ok(decoded);
        }
    }

    if let Some(decoded) = try_decode_yolo_pose_output(
        primary,
        meta,
        class_names,
        conf_threshold,
        iou_threshold,
        max_detections,
        nms_class_agnostic,
        keypoint_confidence_threshold,
    )? {
        return Ok(decoded);
    }

    if let Some(decoded) = try_decode_yolo_obb_output(
        primary,
        meta,
        class_names,
        conf_threshold,
        iou_threshold,
        max_detections,
        nms_class_agnostic,
    )? {
        return Ok(decoded);
    }

    let boxes = decode_yolo_detect_output(
        primary,
        meta,
        class_names,
        conf_threshold,
        iou_threshold,
        max_detections,
        nms_class_agnostic,
    )?;
    Ok(DecodedYoloOutput {
        task: YoloTaskKind::Detect,
        boxes,
        ..Default::default()
    })
}

/// Decode outputs with an explicitly specified task kind, bypassing auto-detection.
fn decode_by_task_kind(
    task: YoloTaskKind,
    outputs: &[OnnxOutputTensor<'_>],
    primary: &OnnxOutputTensor<'_>,
    meta: &LetterboxMeta,
    class_names: &[String],
    conf_threshold: f32,
    iou_threshold: f32,
    max_detections: usize,
    top_k: Option<usize>,
    nms_class_agnostic: bool,
    keypoint_confidence_threshold: f32,
) -> Result<DecodedYoloOutput> {
    match task {
        YoloTaskKind::Classify => decode_yolo_classify_output(
            primary, class_names, conf_threshold, top_k, max_detections,
        ),
        YoloTaskKind::Segment => {
            let proto = find_mask_proto_output(outputs);
            if let Some(proto) = proto {
                try_decode_yolo_segment_output(
                    primary, proto, meta, class_names, conf_threshold,
                    iou_threshold, max_detections, nms_class_agnostic,
                )
                .map(|opt| opt.unwrap_or_else(|| DecodedYoloOutput {
                    task: YoloTaskKind::Segment,
                    ..Default::default()
                }))
            } else {
                Ok(DecodedYoloOutput {
                    task: YoloTaskKind::Segment,
                    ..Default::default()
                })
            }
        }
        YoloTaskKind::Pose => {
            try_decode_yolo_pose_output(
                primary, meta, class_names, conf_threshold,
                iou_threshold, max_detections, nms_class_agnostic,
                keypoint_confidence_threshold,
            )
            .map(|opt| opt.unwrap_or_else(|| DecodedYoloOutput {
                task: YoloTaskKind::Pose,
                ..Default::default()
            }))
        }
        YoloTaskKind::Obb => {
            try_decode_yolo_obb_output(
                primary, meta, class_names, conf_threshold,
                iou_threshold, max_detections, nms_class_agnostic,
            )
            .map(|opt| opt.unwrap_or_else(|| DecodedYoloOutput {
                task: YoloTaskKind::Obb,
                ..Default::default()
            }))
        }
        YoloTaskKind::Detect => {
            let boxes = decode_yolo_detect_output(
                primary, meta, class_names, conf_threshold,
                iou_threshold, max_detections, nms_class_agnostic,
            )?;
            Ok(DecodedYoloOutput {
                task: YoloTaskKind::Detect,
                boxes,
                ..Default::default()
            })
        }
    }
}

fn decode_yolo_detect_output(
    tensor: &OnnxOutputTensor<'_>,
    meta: &LetterboxMeta,
    class_names: &[String],
    conf_threshold: f32,
    iou_threshold: f32,
    max_detections: usize,
    nms_class_agnostic: bool,
) -> Result<Vec<YoloBox>> {
    let layout = prediction_layout(&tensor.shape, 6)?;
    let mut candidates = Vec::new();
    for pred_index in 0..layout.pred_count {
        let prediction = prediction_at(tensor.values, &layout, pred_index);
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
        nms_class_agnostic,
    ))
}

fn decode_yolo_classify_output(
    tensor: &OnnxOutputTensor<'_>,
    class_names: &[String],
    conf_threshold: f32,
    top_k: Option<usize>,
    max_detections: usize,
) -> Result<DecodedYoloOutput> {
    if tensor.values.is_empty() {
        return Err(anyhow!("classification output tensor is empty"));
    }
    let top_k = top_k.unwrap_or(max_detections).max(1);
    let probs = classification_probabilities(tensor.values);
    let mut predictions = probs
        .iter()
        .copied()
        .enumerate()
        .map(|(cls, conf)| YoloClassification {
            cls,
            label: class_names
                .get(cls)
                .cloned()
                .unwrap_or_else(|| cls.to_string()),
            conf,
        })
        .filter(|prediction| prediction.conf >= conf_threshold)
        .collect::<Vec<_>>();
    if predictions.is_empty() {
        if let Some((cls, conf)) = probs
            .iter()
            .copied()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(&right.1))
        {
            predictions.push(YoloClassification {
                cls,
                label: class_names
                    .get(cls)
                    .cloned()
                    .unwrap_or_else(|| cls.to_string()),
                conf,
            });
        }
    }
    predictions.sort_by(|left, right| right.conf.total_cmp(&left.conf));
    predictions.truncate(top_k);
    Ok(DecodedYoloOutput {
        task: YoloTaskKind::Classify,
        probs: predictions,
        ..Default::default()
    })
}

fn try_decode_yolo_segment_output(
    tensor: &OnnxOutputTensor<'_>,
    proto: &OnnxOutputTensor<'_>,
    meta: &LetterboxMeta,
    class_names: &[String],
    conf_threshold: f32,
    iou_threshold: f32,
    max_detections: usize,
    nms_class_agnostic: bool,
) -> Result<Option<DecodedYoloOutput>> {
    let Some(mask_dim) = infer_proto_channels(&proto.shape) else {
        return Ok(None);
    };
    let layout = prediction_layout(&tensor.shape, 6)?;
    let Some(aux_layout) = infer_fixed_extra_layout(layout.attr_count, class_names.len(), mask_dim) else {
        return Ok(None);
    };

    let mut candidates = Vec::new();
    for pred_index in 0..layout.pred_count {
        let prediction = prediction_at(tensor.values, &layout, pred_index);
        if let Some((detection, coeffs)) = decode_segment_prediction(
            &prediction,
            meta,
            class_names,
            conf_threshold,
            aux_layout,
        ) {
            candidates.push((detection, coeffs));
        }
    }

    let selected = non_max_suppression_with_aux(candidates, iou_threshold, max_detections, nms_class_agnostic);
    if selected.is_empty() {
        return Ok(Some(DecodedYoloOutput {
            task: YoloTaskKind::Segment,
            ..Default::default()
        }));
    }

    let mut boxes = Vec::with_capacity(selected.len());
    let mut masks = Vec::with_capacity(selected.len());
    for (detection, coeffs) in selected {
        let mask = decode_mask_from_proto(proto, &coeffs, meta, &detection.xyxy)
            .unwrap_or_else(|_| decode_rect_mask(meta, &detection.xyxy));
        boxes.push(detection);
        masks.push(mask);
    }
    Ok(Some(DecodedYoloOutput {
        task: YoloTaskKind::Segment,
        boxes,
        masks,
        ..Default::default()
    }))
}

fn try_decode_yolo_obb_output(
    tensor: &OnnxOutputTensor<'_>,
    meta: &LetterboxMeta,
    class_names: &[String],
    conf_threshold: f32,
    iou_threshold: f32,
    max_detections: usize,
    nms_class_agnostic: bool,
) -> Result<Option<DecodedYoloOutput>> {
    if !is_probable_obb_output(tensor) {
        return Ok(None);
    }
    let layout = prediction_layout(&tensor.shape, 6)?;
    let Some(aux_layout) = infer_obb_aux_layout(layout.attr_count, class_names.len()) else {
        return Ok(None);
    };

    let mut candidates = Vec::new();
    for pred_index in 0..layout.pred_count {
        let prediction = prediction_at(tensor.values, &layout, pred_index);
        if let Some(obb) = decode_obb_prediction(
            &prediction,
            meta,
            class_names,
            conf_threshold,
            aux_layout,
        ) {
            candidates.push(obb);
        }
    }

    let obb = non_max_suppression_obb(candidates, iou_threshold, max_detections, nms_class_agnostic);
    Ok(Some(DecodedYoloOutput {
        task: YoloTaskKind::Obb,
        obb,
        ..Default::default()
    }))
}

fn try_decode_yolo_pose_output(
    tensor: &OnnxOutputTensor<'_>,
    meta: &LetterboxMeta,
    class_names: &[String],
    conf_threshold: f32,
    iou_threshold: f32,
    max_detections: usize,
    nms_class_agnostic: bool,
    keypoint_confidence_threshold: f32,
) -> Result<Option<DecodedYoloOutput>> {
    let layout = prediction_layout(&tensor.shape, 6)?;
    let Some(aux_layout) = infer_pose_aux_layout(layout.attr_count, class_names.len()) else {
        return Ok(None);
    };

    let mut candidates = Vec::new();
    for pred_index in 0..layout.pred_count {
        let prediction = prediction_at(tensor.values, &layout, pred_index);
        if let Some((detection, keypoints)) = decode_pose_prediction(
            &prediction,
            meta,
            class_names,
            conf_threshold,
            aux_layout,
            keypoint_confidence_threshold,
        ) {
            candidates.push((detection, keypoints));
        }
    }

    let selected = non_max_suppression_with_aux(candidates, iou_threshold, max_detections, nms_class_agnostic);
    let mut boxes = Vec::with_capacity(selected.len());
    let mut keypoints = Vec::with_capacity(selected.len());
    for (detection, points) in selected {
        boxes.push(detection);
        keypoints.push(points);
    }
    Ok(Some(DecodedYoloOutput {
        task: YoloTaskKind::Pose,
        boxes,
        keypoints,
        ..Default::default()
    }))
}

fn select_primary_output<'a>(outputs: &'a [OnnxOutputTensor<'a>]) -> Option<&'a OnnxOutputTensor<'a>> {
    outputs
        .iter()
        .find(|tensor| tensor.shape.len() != 4)
        .or_else(|| outputs.first())
}

fn find_mask_proto_output<'a>(outputs: &'a [OnnxOutputTensor<'a>]) -> Option<&'a OnnxOutputTensor<'a>> {
    outputs.iter().find(|tensor| tensor.shape.len() == 4)
}

fn is_classification_output(tensor: &OnnxOutputTensor<'_>) -> bool {
    match tensor.shape.as_slice() {
        [classes] => *classes > 1,
        [1, classes] => *classes > 1,
        [1, 1, classes] => *classes > 1,
        [1, classes, 1] => *classes > 1,
        _ => false,
    }
}

fn is_probable_obb_output(tensor: &OnnxOutputTensor<'_>) -> bool {
    let name = tensor.name.to_ascii_lowercase();
    name.contains("obb")
        || name.contains("angle")
        || name.contains("rot")
        || name.contains("xywhr")
}

fn prediction_layout(shape: &[i64], min_attr_count: usize) -> Result<PredictionLayout> {
    if shape.len() != 3 || shape[0] != 1 {
        return Err(anyhow!("unsupported yolo output shape: {:?}", shape));
    }
    let dim1 = shape[1].max(1) as usize;
    let dim2 = shape[2].max(1) as usize;
    let (pred_count, attr_count, transposed) = if dim1 > dim2 && dim2 >= min_attr_count {
        (dim2, dim1, true)
    } else {
        (dim1, dim2, false)
    };
    if attr_count < min_attr_count {
        return Err(anyhow!("unsupported yolo attribute count {attr_count} for shape {:?}", shape));
    }
    Ok(PredictionLayout {
        pred_count,
        attr_count,
        transposed,
    })
}

fn prediction_at(values: &[f32], layout: &PredictionLayout, pred_index: usize) -> Vec<f32> {
    if layout.transposed {
        (0..layout.attr_count)
            .map(|attr_index| values[attr_index * layout.pred_count + pred_index])
            .collect()
    } else {
        let start = pred_index * layout.attr_count;
        values[start..start + layout.attr_count].to_vec()
    }
}

fn classification_probabilities(values: &[f32]) -> Vec<f32> {
    if values.is_empty() {
        return Vec::new();
    }
    let all_probabilities = values.iter().all(|value| (0.0..=1.0).contains(value));
    let sum = values.iter().sum::<f32>();
    if all_probabilities && (0.99..=1.01).contains(&sum) {
        return values.to_vec();
    }
    softmax(values)
}

fn softmax(values: &[f32]) -> Vec<f32> {
    if values.is_empty() {
        return Vec::new();
    }
    let max = values
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let exps = values
        .iter()
        .map(|value| (*value - max).exp())
        .collect::<Vec<_>>();
    let sum = exps.iter().sum::<f32>().max(f32::EPSILON);
    exps.into_iter().map(|value| value / sum).collect()
}

fn infer_fixed_extra_layout(
    attr_count: usize,
    preferred_class_count: usize,
    extra_dims: usize,
) -> Option<PredictionAuxLayout> {
    for has_objectness in [true, false] {
        let base = if has_objectness { 5 } else { 4 };
        if attr_count <= base + extra_dims {
            continue;
        }
        let class_count = if preferred_class_count > 0 {
            preferred_class_count
        } else {
            attr_count.checked_sub(base + extra_dims)?
        };
        if attr_count == base + class_count + extra_dims && class_count > 0 {
            return Some(PredictionAuxLayout {
                has_objectness,
                class_count,
                extra_start: base + class_count,
            });
        }
    }
    None
}

fn infer_obb_aux_layout(
    attr_count: usize,
    preferred_class_count: usize,
) -> Option<PredictionAuxLayout> {
    infer_fixed_extra_layout(attr_count, preferred_class_count, 1)
}

fn infer_pose_aux_layout(attr_count: usize, preferred_class_count: usize) -> Option<PredictionAuxLayout> {
    for has_objectness in [true, false] {
        let base = if has_objectness { 5 } else { 4 };
        if attr_count <= base + preferred_class_count + 5 {
            continue;
        }
        let extra_dims = attr_count.checked_sub(base + preferred_class_count)?;
        if extra_dims >= 6 && extra_dims % 3 == 0 {
            return Some(PredictionAuxLayout {
                has_objectness,
                class_count: preferred_class_count,
                extra_start: base + preferred_class_count,
            });
        }
    }
    None
}

fn decode_segment_prediction(
    prediction: &[f32],
    meta: &LetterboxMeta,
    class_names: &[String],
    conf_threshold: f32,
    aux_layout: PredictionAuxLayout,
) -> Option<(YoloBox, Vec<f32>)> {
    let detection = decode_aux_prediction(
        prediction,
        meta,
        class_names,
        conf_threshold,
        aux_layout,
    )?;
    Some((detection, prediction[aux_layout.extra_start..].to_vec()))
}

fn decode_pose_prediction(
    prediction: &[f32],
    meta: &LetterboxMeta,
    class_names: &[String],
    conf_threshold: f32,
    aux_layout: PredictionAuxLayout,
    keypoint_confidence_threshold: f32,
) -> Option<(YoloBox, Vec<YoloKeypoint>)> {
    let detection = decode_aux_prediction(
        prediction,
        meta,
        class_names,
        conf_threshold,
        aux_layout,
    )?;
    let keypoints: Vec<YoloKeypoint> = prediction[aux_layout.extra_start..]
        .chunks_exact(3)
        .map(|chunk| {
            let [x, y] = restore_point(chunk[0], chunk[1], meta);
            YoloKeypoint {
                x,
                y,
                conf: normalize_confidence(chunk[2]),
            }
        })
        .filter(|kp| kp.conf >= keypoint_confidence_threshold)
        .collect();
    Some((detection, keypoints))
}

fn decode_obb_prediction(
    prediction: &[f32],
    meta: &LetterboxMeta,
    class_names: &[String],
    conf_threshold: f32,
    aux_layout: PredictionAuxLayout,
) -> Option<YoloObb> {
    let detection = decode_aux_prediction(
        prediction,
        meta,
        class_names,
        conf_threshold,
        aux_layout,
    )?;
    let angle = normalize_obb_angle(*prediction.get(aux_layout.extra_start)?);
    let center = restore_point(prediction[0], prediction[1], meta);
    let width = (prediction[2] / meta.scale).max(0.0);
    let height = (prediction[3] / meta.scale).max(0.0);
    let corners = oriented_box_corners(center, width, height, angle);
    let xyxy = corners_to_xyxy(&corners);
    Some(YoloObb {
        xywhr: [center[0], center[1], width, height, angle],
        corners,
        xyxy,
        conf: detection.conf,
        cls: detection.cls,
        label: detection.label,
    })
}

fn decode_aux_prediction(
    prediction: &[f32],
    meta: &LetterboxMeta,
    class_names: &[String],
    conf_threshold: f32,
    aux_layout: PredictionAuxLayout,
) -> Option<YoloBox> {
    if prediction.len() < aux_layout.extra_start {
        return None;
    }
    let class_start = if aux_layout.has_objectness { 5 } else { 4 };
    let objectness = if aux_layout.has_objectness {
        normalize_confidence(prediction[4])
    } else {
        1.0
    };
    let class_end = class_start + aux_layout.class_count;
    let class_scores = prediction.get(class_start..class_end)?;
    let (cls, class_score) = class_scores
        .iter()
        .copied()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(&right.1))?;
    let conf = normalize_confidence(class_score) * objectness;
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

fn infer_proto_channels(shape: &[i64]) -> Option<usize> {
    if shape.len() != 4 {
        return None;
    }
    let c_first = shape[1].max(1) as usize;
    let c_last = shape[3].max(1) as usize;
    Some(c_first.min(c_last))
}

fn decode_rect_mask(meta: &LetterboxMeta, xyxy: &[f32; 4]) -> YoloMask {
    let width = meta.orig_w;
    let height = meta.orig_h;
    let mut data = vec![0_u8; width as usize * height as usize];
    let x1 = xyxy[0].floor().clamp(0.0, width as f32) as u32;
    let y1 = xyxy[1].floor().clamp(0.0, height as f32) as u32;
    let x2 = xyxy[2].ceil().clamp(0.0, width as f32) as u32;
    let y2 = xyxy[3].ceil().clamp(0.0, height as f32) as u32;
    for y in y1..y2 {
        for x in x1..x2 {
            let index = (y * width + x) as usize;
            if let Some(cell) = data.get_mut(index) {
                *cell = 255;
            }
        }
    }
    YoloMask { width, height, data }
}

fn decode_mask_from_proto(
    proto: &OnnxOutputTensor<'_>,
    coeffs: &[f32],
    meta: &LetterboxMeta,
    xyxy: &[f32; 4],
) -> Result<YoloMask> {
    if proto.shape.len() != 4 {
        return Err(anyhow!("unsupported proto shape: {:?}", proto.shape));
    }
    let width = meta.orig_w;
    let height = meta.orig_h;
    let mut data = vec![0_u8; width as usize * height as usize];
    if coeffs.is_empty() {
        return Ok(YoloMask { width, height, data });
    }

    let (channels, proto_h, proto_w, nchw) = if proto.shape[1].max(1) as usize == coeffs.len() {
        (
            proto.shape[1].max(1) as usize,
            proto.shape[2].max(1) as usize,
            proto.shape[3].max(1) as usize,
            true,
        )
    } else if proto.shape[3].max(1) as usize == coeffs.len() {
        (
            proto.shape[3].max(1) as usize,
            proto.shape[1].max(1) as usize,
            proto.shape[2].max(1) as usize,
            false,
        )
    } else {
        return Err(anyhow!("proto channel count does not match coeffs"));
    };

    let input_w = (meta.orig_w as f32 * meta.scale + meta.pad_x * 2.0).max(1.0);
    let input_h = (meta.orig_h as f32 * meta.scale + meta.pad_y * 2.0).max(1.0);
    let x1 = xyxy[0].floor().clamp(0.0, width as f32) as u32;
    let y1 = xyxy[1].floor().clamp(0.0, height as f32) as u32;
    let x2 = xyxy[2].ceil().clamp(0.0, width as f32) as u32;
    let y2 = xyxy[3].ceil().clamp(0.0, height as f32) as u32;

    for y in y1..y2 {
        for x in x1..x2 {
            let x_input = x as f32 * meta.scale + meta.pad_x;
            let y_input = y as f32 * meta.scale + meta.pad_y;
            let px = ((x_input / input_w) * proto_w as f32)
                .floor()
                .clamp(0.0, (proto_w.saturating_sub(1)) as f32) as usize;
            let py = ((y_input / input_h) * proto_h as f32)
                .floor()
                .clamp(0.0, (proto_h.saturating_sub(1)) as f32) as usize;
            let mut logit = 0.0_f32;
            for channel in 0..channels {
                let proto_index = if nchw {
                    channel * proto_h * proto_w + py * proto_w + px
                } else {
                    py * proto_w * channels + px * channels + channel
                };
                logit += coeffs[channel] * proto.values.get(proto_index).copied().unwrap_or(0.0);
            }
            if normalize_confidence(logit) >= 0.5 {
                let index = (y * width + x) as usize;
                if let Some(cell) = data.get_mut(index) {
                    *cell = 255;
                }
            }
        }
    }

    Ok(YoloMask { width, height, data })
}

fn restore_point(x: f32, y: f32, meta: &LetterboxMeta) -> [f32; 2] {
    [
        ((x - meta.pad_x) / meta.scale).clamp(0.0, meta.orig_w as f32),
        ((y - meta.pad_y) / meta.scale).clamp(0.0, meta.orig_h as f32),
    ]
}

fn normalize_obb_angle(value: f32) -> f32 {
    // If the value is already within a reasonable angle range, keep it as-is.
    if value.abs() <= std::f32::consts::PI * 2.0 {
        return value;
    }
    // For values that look like raw logits (large magnitude), apply sigmoid
    // and then scale to [0, π]. This matches the Ultralytics convention where
    // the angle output is sigmoid(angle_raw) * π when the raw value is unbounded.
    normalize_confidence(value) * std::f32::consts::PI
}

fn oriented_box_corners(center: [f32; 2], width: f32, height: f32, angle: f32) -> [[f32; 2]; 4] {
    let half_w = width * 0.5;
    let half_h = height * 0.5;
    let cos = angle.cos();
    let sin = angle.sin();
    let offsets = [
        [-half_w, -half_h],
        [half_w, -half_h],
        [half_w, half_h],
        [-half_w, half_h],
    ];
    offsets.map(|offset| {
        [
            center[0] + offset[0] * cos - offset[1] * sin,
            center[1] + offset[0] * sin + offset[1] * cos,
        ]
    })
}

fn corners_to_xyxy(corners: &[[f32; 2]; 4]) -> [f32; 4] {
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for point in corners {
        min_x = min_x.min(point[0]);
        min_y = min_y.min(point[1]);
        max_x = max_x.max(point[0]);
        max_y = max_y.max(point[1]);
    }
    [min_x, min_y, max_x, max_y]
}

fn normalize_confidence(value: f32) -> f32 {
    if (0.0..=1.0).contains(&value) {
        value
    } else {
        1.0 / (1.0 + (-value).exp())
    }
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

    let (objectness, class_scores) = split_detection_prediction(prediction, class_names.len())?;
    let (cls, class_score) = class_scores
        .iter()
        .copied()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(&right.1))?;
    let conf = normalize_confidence(class_score) * normalize_confidence(objectness);
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
    detections: Vec<YoloBox>,
    iou_threshold: f32,
    max_detections: usize,
    nms_class_agnostic: bool,
) -> Vec<YoloBox> {
    let mut selected = Vec::new();
    let mut suppressed = vec![false; detections.len()];
    for index in 0..detections.len() {
        if suppressed[index] || selected.len() >= max_detections {
            continue;
        }
        let current = detections[index].clone();
        for candidate_index in (index + 1)..detections.len() {
            if suppressed[candidate_index] {
                continue;
            }
            let same_class = detections[candidate_index].cls == current.cls;
            if (nms_class_agnostic || same_class)
                && intersection_over_union(&current.xyxy, &detections[candidate_index].xyxy) >= iou_threshold
            {
                suppressed[candidate_index] = true;
            }
        }
        selected.push(current);
    }
    selected
}

fn non_max_suppression_obb(
    detections: Vec<YoloObb>,
    iou_threshold: f32,
    max_detections: usize,
    nms_class_agnostic: bool,
) -> Vec<YoloObb> {
    let mut selected = Vec::new();
    let mut suppressed = vec![false; detections.len()];
    for index in 0..detections.len() {
        if suppressed[index] || selected.len() >= max_detections {
            continue;
        }
        let current = detections[index].clone();
        for candidate_index in (index + 1)..detections.len() {
            if suppressed[candidate_index] {
                continue;
            }
            let same_class = detections[candidate_index].cls == current.cls;
            if nms_class_agnostic || same_class {
                let iou = rotated_iou(&current.corners, &detections[candidate_index].corners);
                if iou >= iou_threshold {
                    suppressed[candidate_index] = true;
                }
            }
        }
        selected.push(current);
    }
    selected
}

fn non_max_suppression_with_aux<T: Clone>(
    detections: Vec<(YoloBox, T)>,
    iou_threshold: f32,
    max_detections: usize,
    nms_class_agnostic: bool,
) -> Vec<(YoloBox, T)> {
    let mut selected = Vec::new();
    let mut suppressed = vec![false; detections.len()];
    for index in 0..detections.len() {
        if suppressed[index] || selected.len() >= max_detections {
            continue;
        }
        let current = detections[index].clone();
        for candidate_index in (index + 1)..detections.len() {
            if suppressed[candidate_index] {
                continue;
            }
            let same_class = detections[candidate_index].0.cls == current.0.cls;
            if (nms_class_agnostic || same_class)
                && intersection_over_union(&current.0.xyxy, &detections[candidate_index].0.xyxy)
                    >= iou_threshold
            {
                suppressed[candidate_index] = true;
            }
        }
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

/// Compute IoU for two oriented (rotated) bounding boxes using their corner polygons.
/// Uses the Sutherland-Hodgman algorithm to clip one polygon against the other,
/// then computes the intersection area from the clipped polygon.
fn rotated_iou(corners_a: &[[f32; 2]; 4], corners_b: &[[f32; 2]; 4]) -> f32 {
    let area_a = polygon_area(corners_a);
    let area_b = polygon_area(corners_b);
    if area_a <= 0.0 || area_b <= 0.0 {
        return 0.0;
    }
    let inter = polygon_intersection_area(corners_a, corners_b);
    let union = area_a + area_b - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

/// Compute the signed area of a convex polygon using the shoelace formula.
fn polygon_area(vertices: &[[f32; 2]; 4]) -> f32 {
    let n = vertices.len();
    let mut area = 0.0_f32;
    for i in 0..n {
        let j = (i + 1) % n;
        area += vertices[i][0] * vertices[j][1];
        area -= vertices[j][0] * vertices[i][1];
    }
    area.abs() * 0.5
}

/// Compute the intersection area of two convex polygons using Sutherland-Hodgman clipping.
/// Clips polygon `subject` against each edge of polygon `clip`, then computes the
/// area of the resulting intersection polygon.
fn polygon_intersection_area(
    subject: &[[f32; 2]; 4],
    clip: &[[f32; 2]; 4],
) -> f32 {
    let mut output: Vec<[f32; 2]> = subject.to_vec();
    let clip_len = clip.len();

    for i in 0..clip_len {
        if output.is_empty() {
            return 0.0;
        }
        let input = std::mem::take(&mut output);
        let edge_start = clip[i];
        let edge_end = clip[(i + 1) % clip_len];

        for j in 0..input.len() {
            let current = input[j];
            let previous = input[(j + input.len() - 1) % input.len()];

            let current_inside = is_left_of_edge(edge_start, edge_end, current);
            let previous_inside = is_left_of_edge(edge_start, edge_end, previous);

            match (previous_inside, current_inside) {
                (true, true) => {
                    output.push(current);
                }
                (true, false) => {
                    if let Some(pt) = line_intersection(previous, current, edge_start, edge_end) {
                        output.push(pt);
                    }
                }
                (false, true) => {
                    if let Some(pt) = line_intersection(previous, current, edge_start, edge_end) {
                        output.push(pt);
                    }
                    output.push(current);
                }
                (false, false) => {}
            }
        }
    }

    if output.len() < 3 {
        return 0.0;
    }
    shoelace_area(&output)
}

/// Test if point is on the left side (inside) of the directed edge from `start` to `end`.
/// For a counter-clockwise polygon, "left" is inside. For clockwise, we use >= 0
/// to handle both orientations.
fn is_left_of_edge(edge_start: [f32; 2], edge_end: [f32; 2], point: [f32; 2]) -> bool {
    let cross = (edge_end[0] - edge_start[0]) * (point[1] - edge_start[1])
        - (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0]);
    cross >= 0.0
}

/// Compute the intersection point of two line segments (p1,p2) and (p3,p4).
/// Returns None if the lines are parallel.
fn line_intersection(
    p1: [f32; 2],
    p2: [f32; 2],
    p3: [f32; 2],
    p4: [f32; 2],
) -> Option<[f32; 2]> {
    let denom = (p1[0] - p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0] - p4[0]);
    if denom.abs() < 1e-10 {
        return None;
    }
    let t = ((p1[0] - p3[0]) * (p3[1] - p4[1]) - (p1[1] - p3[1]) * (p3[0] - p4[0])) / denom;
    Some([
        p1[0] + t * (p2[0] - p1[0]),
        p1[1] + t * (p2[1] - p1[1]),
    ])
}

/// Shoelace formula for polygon area (variable vertex count).
fn shoelace_area(vertices: &[[f32; 2]]) -> f32 {
    let n = vertices.len();
    if n < 3 {
        return 0.0;
    }
    let mut area = 0.0_f32;
    for i in 0..n {
        let j = (i + 1) % n;
        area += vertices[i][0] * vertices[j][1];
        area -= vertices[j][0] * vertices[i][1];
    }
    area.abs() * 0.5
}

fn split_detection_prediction<'a>(
    prediction: &'a [f32],
    preferred_class_count: usize,
) -> Option<(f32, &'a [f32])> {
    if prediction.len() <= 4 {
        return None;
    }
    if preferred_class_count > 0 {
        if prediction.len() == preferred_class_count + 5 {
            return Some((prediction[4], &prediction[5..]));
        }
        if prediction.len() == preferred_class_count + 4 {
            return Some((1.0, &prediction[4..]));
        }
    }

    let with_objectness = if prediction.len() > 5 {
        Some((prediction[4], &prediction[5..]))
    } else {
        None
    };
    let without_objectness = Some((1.0, &prediction[4..]));

    match (with_objectness, without_objectness) {
        (Some((obj, scores)), Some((fallback_obj, fallback_scores))) => {
            let with_score = scores.iter().copied().map(normalize_confidence).fold(0.0_f32, f32::max)
                * normalize_confidence(obj);
            let without_score = fallback_scores
                .iter()
                .copied()
                .map(normalize_confidence)
                .fold(0.0_f32, f32::max)
                * fallback_obj;
            if with_score >= without_score {
                Some((obj, scores))
            } else {
                Some((fallback_obj, fallback_scores))
            }
        }
        (Some(value), None) => Some(value),
        (None, Some(value)) => Some(value),
        (None, None) => None,
    }
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
        if ch.is_ascii() {
            // Render ASCII characters using 8x8 bitmap font
            if let Some(glyph) = BASIC_FONTS.get(ch) {
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
        } else {
            // Non-ASCII characters: draw a filled 8x8 block as placeholder
            for row in 0..8u32 {
                let yy = y.saturating_add(row);
                if yy >= height {
                    break;
                }
                for col in 0..8u32 {
                    let xx = cursor_x.saturating_add(col);
                    if xx < width {
                        image.put_pixel(xx, yy, color);
                    }
                }
            }
            cursor_x = cursor_x.saturating_add(8);
        }
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
        let result = backend.predict(&item, config, options.top_k)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_meta() -> LetterboxMeta {
        LetterboxMeta {
            scale: 1.0,
            pad_x: 0.0,
            pad_y: 0.0,
            orig_w: 32,
            orig_h: 32,
        }
    }

    #[test]
    fn classify_decode_uses_top_k() {
        let logits = vec![0.1_f32, 4.0, 2.5, 1.0];
        let output = OnnxOutputTensor {
            name: "output0".to_string(),
            shape: vec![1, 4],
            values: &logits,
        };
        let decoded = decode_yolo_outputs(
            &[output],
            &dummy_meta(),
            &["a".to_string(), "b".to_string(), "c".to_string(), "d".to_string()],
            0.0,
            0.45,
            10,
            Some(2),
            None,
            false,
            0.1,
        )
        .expect("classification decode should succeed");
        assert_eq!(decoded.task, YoloTaskKind::Classify);
        assert_eq!(decoded.probs.len(), 2);
        assert_eq!(decoded.probs[0].label, "b");
        assert_eq!(decoded.probs[1].label, "c");
    }

    #[test]
    fn classify_decode_keeps_best_prediction_when_threshold_filters_all() {
        let logits = vec![0.1_f32, 0.2, 0.3];
        let output = OnnxOutputTensor {
            name: "output0".to_string(),
            shape: vec![3],
            values: &logits,
        };
        let decoded = decode_yolo_outputs(
            &[output],
            &dummy_meta(),
            &["x".to_string(), "y".to_string(), "z".to_string()],
            0.99,
            0.45,
            10,
            Some(1),
            None,
            false,
            0.1,
        )
        .expect("classification decode should keep best prediction");
        assert_eq!(decoded.task, YoloTaskKind::Classify);
        assert_eq!(decoded.probs.len(), 1);
        assert_eq!(decoded.probs[0].label, "z");
    }

    #[test]
    fn obb_decode_returns_oriented_boxes() {
        let predictions = vec![
            16.0_f32, 16.0, 8.0, 4.0, 0.95, 0.3,
            8.0_f32, 8.0, 2.0, 2.0, 0.10, 0.1,
        ];
        let output = OnnxOutputTensor {
            name: "obb".to_string(),
            shape: vec![1, 2, 6],
            values: &predictions,
        };
        let decoded = decode_yolo_outputs(
            &[output],
            &dummy_meta(),
            &["plane".to_string()],
            0.25,
            0.45,
            10,
            None,
            None,
            false,
            0.1,
        )
        .expect("obb decode should succeed");
        assert_eq!(decoded.task, YoloTaskKind::Obb);
        assert_eq!(decoded.obb.len(), 1);
        assert_eq!(decoded.obb[0].label, "plane");
        assert!(decoded.obb[0].xywhr[2] > 0.0);
        assert!(decoded.obb[0].xywhr[3] > 0.0);
    }

    #[test]
    fn task_kind_override_bypasses_auto_detection() {
        // A classification-shaped output, but we force Detect mode
        let logits = vec![0.1_f32, 4.0, 2.5, 1.0];
        let output = OnnxOutputTensor {
            name: "output0".to_string(),
            shape: vec![1, 4],
            values: &logits,
        };
        let decoded = decode_yolo_outputs(
            &[output],
            &dummy_meta(),
            &["a".to_string(), "b".to_string(), "c".to_string(), "d".to_string()],
            0.0,
            0.45,
            10,
            None,
            Some(YoloTaskKind::Detect),
            false,
            0.1,
        )
        .expect("detect with task override should succeed");
        assert_eq!(decoded.task, YoloTaskKind::Detect);
    }

    #[test]
    fn nms_class_agnostic_suppresses_across_classes() {
        // Two boxes with different classes but high IoU
        let box1 = YoloBox {
            xyxy: [0.0, 0.0, 10.0, 10.0],
            xywh: [5.0, 5.0, 10.0, 10.0],
            conf: 0.9,
            cls: 0,
            label: "a".to_string(),
        };
        let box2 = YoloBox {
            xyxy: [1.0, 1.0, 11.0, 11.0],
            xywh: [6.0, 6.0, 10.0, 10.0],
            conf: 0.8,
            cls: 1,
            label: "b".to_string(),
        };
        // Class-aware: both survive
        let result_aware = non_max_suppression(
            vec![box1.clone(), box2.clone()],
            0.5,
            100,
            false,
        );
        assert_eq!(result_aware.len(), 2);
        // Class-agnostic: second is suppressed
        let result_agnostic = non_max_suppression(
            vec![box1, box2],
            0.5,
            100,
            true,
        );
        assert_eq!(result_agnostic.len(), 1);
    }

    #[test]
    fn keypoint_confidence_threshold_filters_low_confidence() {
        let predictions: Vec<f32> = vec![
            16.0, 16.0, 8.0, 4.0, // cx, cy, w, h
            0.95, // objectness
            0.9,  // class score (1 class)
            // 2 keypoints: x, y, conf
            10.0, 10.0, 0.8,
            20.0, 20.0, 0.05,
        ];
        let output = OnnxOutputTensor {
            name: "output0".to_string(),
            shape: vec![1, 1, 11],
            values: &predictions,
        };
        let decoded = decode_yolo_outputs(
            &[output],
            &dummy_meta(),
            &["person".to_string()],
            0.25,
            0.45,
            10,
            None,
            Some(YoloTaskKind::Pose),
            false,
            0.1, // keypoint_confidence_threshold = 0.1
        )
        .expect("pose decode should succeed");
        // The second keypoint (conf=0.05) should be filtered out
        assert_eq!(decoded.keypoints.len(), 1);
        assert_eq!(decoded.keypoints[0].len(), 1);
        assert!(decoded.keypoints[0][0].conf >= 0.1);
    }

    #[test]
    fn rotated_iou_identical_boxes() {
        let corners = [
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 5.0],
            [0.0, 5.0],
        ];
        let iou = rotated_iou(&corners, &corners);
        assert!((iou - 1.0).abs() < 1e-4, "identical boxes should have IoU=1.0, got {iou}");
    }

    #[test]
    fn rotated_iou_non_overlapping() {
        let corners_a = [
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 5.0],
            [0.0, 5.0],
        ];
        let corners_b = [
            [20.0, 0.0],
            [30.0, 0.0],
            [30.0, 5.0],
            [20.0, 5.0],
        ];
        let iou = rotated_iou(&corners_a, &corners_b);
        assert!(iou < 0.01, "non-overlapping boxes should have IoU≈0.0, got {iou}");
    }

    #[test]
    fn rotated_iou_partial_overlap() {
        let corners_a = [
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0],
        ];
        let corners_b = [
            [5.0, 0.0],
            [15.0, 0.0],
            [15.0, 10.0],
            [5.0, 10.0],
        ];
        let iou = rotated_iou(&corners_a, &corners_b);
        // Intersection = 5*10 = 50, union = 100+100-50 = 150, IoU = 50/150 ≈ 0.333
        assert!((iou - 0.333).abs() < 0.02, "partial overlap should have IoU≈0.333, got {iou}");
    }

    #[test]
    fn rotated_iou_rotated_boxes_lower_than_axis_aligned() {
        // Two boxes at 45° that share the same center but different sizes.
        // Their axis-aligned bounding boxes overlap heavily, but rotated IoU should
        // be much lower than axis-aligned IoU.
        use super::oriented_box_corners;
        let corners_a = oriented_box_corners([50.0, 50.0], 40.0, 10.0, std::f32::consts::FRAC_PI_4);
        let corners_b = oriented_box_corners([50.0, 50.0], 40.0, 10.0, 0.0);
        let rot_iou = rotated_iou(&corners_a, &corners_b);

        // Compute axis-aligned IoU for comparison
        let xyxy_a = super::corners_to_xyxy(&corners_a);
        let xyxy_b = super::corners_to_xyxy(&corners_b);
        let aa_iou = intersection_over_union(&xyxy_a, &xyxy_b);

        assert!(
            rot_iou < aa_iou,
            "rotated IoU ({rot_iou}) should be less than axis-aligned IoU ({aa_iou}) for rotated boxes"
        );
    }

    #[test]
    fn obb_nms_uses_rotated_iou() {
        // Two OBBs with same class, same center, different rotation.
        // Axis-aligned IoU is high, but rotated IoU is low.
        // With rotated IoU, NMS should NOT suppress the second box.
        use super::oriented_box_corners;
        let corners_a = oriented_box_corners([50.0, 50.0], 40.0, 10.0, 0.0);
        let corners_b = oriented_box_corners([50.0, 50.0], 40.0, 10.0, std::f32::consts::FRAC_PI_4);

        let obb_a = YoloObb {
            xywhr: [50.0, 50.0, 40.0, 10.0, 0.0],
            corners: corners_a,
            xyxy: super::corners_to_xyxy(&corners_a),
            conf: 0.9,
            cls: 0,
            label: "plane".to_string(),
        };
        let obb_b = YoloObb {
            xywhr: [50.0, 50.0, 40.0, 10.0, std::f32::consts::FRAC_PI_4],
            corners: corners_b,
            xyxy: super::corners_to_xyxy(&corners_b),
            conf: 0.8,
            cls: 0,
            label: "plane".to_string(),
        };

        // With IoU threshold 0.5, both should survive because rotated IoU is low
        let result = non_max_suppression_obb(
            vec![obb_a, obb_b],
            0.5,
            100,
            false,
        );
        assert_eq!(result.len(), 2, "rotated boxes at different angles should both survive NMS");
    }
}
