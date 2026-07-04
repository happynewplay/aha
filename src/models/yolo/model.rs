#[cfg(feature = "ffmpeg")]
use std::sync::OnceLock;
use std::{
    borrow::Cow,
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Instant,
};

use anyhow::{Result, anyhow};
#[cfg(feature = "ffmpeg")]
use ffmpeg_next as ffmpeg;
use font8x8::{BASIC_FONTS, UnicodeFonts};
use glob::glob;
use image::{DynamicImage, Rgb, RgbImage, Rgba, RgbaImage, imageops};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[cfg(feature = "onnx-runtime")]
use crate::models::common::onnx::create_session;
use crate::{
    models::{ArtifactKind, LoadSpec},
    utils::{get_file_path, img_utils::load_image_from_url},
};

use super::config::{YoloConfig, YoloTaskKind, deserialize_arc_slice, serialize_arc_slice};

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
    /// Bounding box of non-zero pixels as (y_min, y_max, x_min, x_max).
    /// Pre-computed at mask construction time so that rendering can skip
    /// the full-image scan for the bounding box.
    #[serde(skip, default)]
    pub bbox: (u32, u32, u32, u32),
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

#[derive(Debug, Clone, Serialize, Deserialize)]
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
    #[serde(
        serialize_with = "serialize_arc_slice",
        deserialize_with = "deserialize_arc_slice"
    )]
    pub names: Arc<[String]>,
    pub speed: YoloSpeed,
    pub width: u32,
    pub height: u32,
    #[serde(skip, default)]
    pub orig_img: Option<Arc<DynamicImage>>,
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
                annotation["num_keypoints"] =
                    json!(points.iter().filter(|point| point.conf > 0.0).count());
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
        let w = self.width.max(1) as f32;
        let h = self.height.max(1) as f32;
        if !self.probs.is_empty() {
            for prediction in &self.probs {
                lines.push(format!(
                    "{} {:.6} {}",
                    prediction.cls, prediction.conf, prediction.label
                ));
            }
        } else if !self.obb.is_empty() {
            // Ultralytics OBB format: cls x_center y_center width height angle conf
            // with x/y/w/h normalized by image dimensions
            for obb in &self.obb {
                lines.push(format!(
                    "{} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}",
                    obb.cls,
                    obb.xywhr[0] / w,
                    obb.xywhr[1] / h,
                    obb.xywhr[2] / w,
                    obb.xywhr[3] / h,
                    obb.xywhr[4],
                    obb.conf,
                ));
            }
        } else {
            // Ultralytics Detect/Pose/Segment format:
            // cls x_center y_center width height conf [kpts...] [mask...]
            // with x/y/w/h normalized by image dimensions
            for (index, detection) in self.boxes.iter().enumerate() {
                let mut fields = vec![
                    detection.cls.to_string(),
                    format!("{:.6}", detection.xywh[0] / w),
                    format!("{:.6}", detection.xywh[1] / h),
                    format!("{:.6}", detection.xywh[2] / w),
                    format!("{:.6}", detection.xywh[3] / h),
                    format!("{:.6}", detection.conf),
                ];
                if let Some(points) = self.keypoints.get(index)
                    && !points.is_empty()
                {
                    fields.push("kpts".to_string());
                    for point in points {
                        fields.push(format!("{:.6}", point.x / w));
                        fields.push(format!("{:.6}", point.y / h));
                        fields.push(format!("{:.6}", point.conf));
                    }
                }
                if let Some(mask) = self.masks.get(index) {
                    let contour = sample_mask_contour_points(mask, 4, 128);
                    if !contour.is_empty() {
                        fields.push("mask".to_string());
                        for [x, y] in contour {
                            fields.push(format!("{:.6}", x as f32 / w));
                            fields.push(format!("{:.6}", y as f32 / h));
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
    /// Chunk size for distributing work across parallel workers.
    /// Each worker receives up to `chunk_size` images at a time.
    /// This does **not** control the ONNX batch dimension (always 1).
    pub chunk_size: usize,
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
            chunk_size: 16,
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
    let (y_min, y_max, x_min, x_max) = mask.bbox;
    if y_max == 0 && x_max == 0 {
        // Empty mask — nothing to overlay.
        return;
    }
    let alpha_factor = alpha as f32 / 255.0;
    let img_width = image.width() as usize;
    let mask_width = mask.width as usize;
    let row_range_x = x_min as usize..x_max as usize;
    // Get a flat mutable slice of the image pixels for row-level access.
    let img_raw = image.as_flat_samples_mut().samples;
    for y in y_min as usize..y_max as usize {
        let mask_row_start = y * mask_width;
        let img_row_start = y * img_width;
        for x in row_range_x.clone() {
            let mask_value = mask.data.get(mask_row_start + x).copied().unwrap_or(0);
            if mask_value == 0 {
                continue;
            }
            let weight = (mask_value as f32 / 255.0) * alpha_factor;
            let px = img_row_start + x;
            for ch in 0..3 {
                let offset = px * 4 + ch;
                if let Some(cell) = img_raw.get_mut(offset) {
                    *cell = ((*cell as f32 * (1.0 - weight)) + (color.0[ch] as f32 * weight))
                        .round()
                        .clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
}

/// Compute the bounding box of non-zero pixels in `data` within the given
/// row/column bounds. Returns (y_min, y_max, x_min, x_max) for range iteration,
/// or (0, 0, 0, 0) if no non-zero pixel is found.
fn compute_mask_bbox(
    data: &[u8],
    width: u32,
    x1: u32,
    y1: u32,
    x2: u32,
    y2: u32,
) -> (u32, u32, u32, u32) {
    let mut by_min = y2;
    let mut by_max = y1;
    let mut bx_min = x2;
    let mut bx_max = x1;
    for y in y1..y2 {
        for x in x1..x2 {
            if data.get((y * width + x) as usize).copied().unwrap_or(0) != 0 {
                by_min = by_min.min(y);
                by_max = by_max.max(y + 1);
                bx_min = bx_min.min(x);
                bx_max = bx_max.max(x + 1);
            }
        }
    }
    if by_max <= by_min || bx_max <= bx_min {
        (0, 0, 0, 0)
    } else {
        (by_min, by_max, bx_min, bx_max)
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
        draw_line(
            image,
            obb.corners[index],
            obb.corners[(index + 1) % 4],
            color,
        );
    }
    let anchor_x = obb.corners[0][0].max(0.0) as u32;
    let anchor_y = obb.corners[0][1].max(0.0) as u32;
    draw_label(
        image,
        &format!("{} {:.2}", obb.label, obb.conf),
        anchor_x,
        anchor_y,
        color,
    );
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
    let (y_min, y_max, x_min, x_max) = mask.bbox;
    if y_max == 0 && x_max == 0 {
        return Vec::new();
    }
    for y in y_min..y_max {
        for x in x_min..x_max {
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
    use std::collections::HashSet;
    let mut seen = HashSet::<usize>::new();
    let mut categories = Vec::<(usize, String)>::new();
    for result in results {
        for (index, name) in result.names.iter().enumerate() {
            if seen.insert(index) {
                categories.push((index, name.clone()));
            }
        }
        for detection in &result.boxes {
            if seen.insert(detection.cls) {
                categories.push((detection.cls, detection.label.clone()));
            }
        }
        for obb in &result.obb {
            if seen.insert(obb.cls) {
                categories.push((obb.cls, obb.label.clone()));
            }
        }
        for prediction in &result.probs {
            if seen.insert(prediction.cls) {
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
    /// Pool of pre-loaded ONNX backends for parallel inference.
    /// After `predict_parallel` finishes, backends are returned here
    /// so subsequent calls can reuse them instead of reloading from disk.
    backend_pool: Mutex<Vec<YoloOnnxBackend>>,
}

impl YoloModel {
    pub fn init_from_spec(spec: &LoadSpec) -> Result<Self> {
        Self::init_with_config(spec, YoloConfig::default())
    }

    pub fn init_with_config(spec: &LoadSpec, config: YoloConfig) -> Result<Self> {
        let config = config.validate();
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
                    backend_pool: Mutex::new(Vec::new()),
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
                self.predict_sequential(
                    sources,
                    options.top_k,
                    options.stop_flag.as_ref().map(|v| v.as_ref()),
                )?
            } else {
                self.predict_parallel(
                    sources,
                    workers,
                    options.chunk_size.max(1),
                    options.top_k,
                    options.stop_flag.clone(),
                )?
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
            annotations
                .extend(result.to_coco_annotations((index + 1) as u64, &mut next_annotation_id));
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
        top_k: Option<usize>,
        stop_flag: Option<&AtomicBool>,
    ) -> Result<Vec<YoloResults>> {
        // NOTE: Each image is inferred individually with batch_size=1 to the ONNX
        // session because the model input shape is fixed at [1,3,H,W].
        let mut results = Vec::with_capacity(sources.len());
        for item in sources {
            if stop_flag.is_some_and(|f| f.load(Ordering::Relaxed)) {
                break;
            }
            results.push(self.backend.predict(&item, &self.config, top_k)?);
        }
        Ok(results)
    }

    fn predict_parallel(
        &self,
        sources: Vec<ResolvedImageSource>,
        workers: usize,
        chunk_size: usize,
        top_k: Option<usize>,
        stop_flag: Option<Arc<AtomicBool>>,
    ) -> Result<Vec<YoloResults>> {
        let indexed_sources = sources.into_iter().enumerate().collect::<Vec<_>>();
        let actual_chunk = chunk_size.max(indexed_sources.len().div_ceil(workers).max(1));
        let grouped = indexed_sources
            .chunks(actual_chunk)
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>();

        // Try to reuse backends from the pool first, only creating new ones
        // when the pool doesn't have enough. If the mutex is poisoned (a
        // previous thread panicked while holding the lock), recover it but
        // discard the stale backends — they may be in an inconsistent state.
        let needed = grouped.len();
        let mut pool = match self.backend_pool.lock() {
            Ok(guard) => guard,
            Err(e) => {
                // Poisoned mutex: recover access but clear stale data
                let mut guard = e.into_inner();
                guard.clear();
                guard
            }
        };
        let reuse_count = needed.min(pool.len());
        let mut backends: Vec<YoloOnnxBackend> = pool.drain(..reuse_count).collect();
        drop(pool); // Release lock before expensive ONNX session creation.

        while backends.len() < needed {
            backends.push(YoloOnnxBackend::load(&self.onnx_path, &self.config)?);
        }

        let mut joined = Vec::<(usize, YoloResults)>::new();
        let used_backends = thread::scope(|scope| -> Result<Vec<YoloOnnxBackend>> {
            let mut handles = Vec::with_capacity(grouped.len());
            let mut backend_iter = backends.into_iter();
            let stop = stop_flag.clone();
            for group in grouped.into_iter() {
                let mut backend = backend_iter
                    .next()
                    .ok_or_else(|| anyhow!("insufficient ONNX backends for parallel inference"))?;
                let config = self.config.clone();
                let top_k = top_k;
                let stop = stop.clone();
                handles.push(scope.spawn(
                    move || -> Result<(Vec<(usize, YoloResults)>, YoloOnnxBackend)> {
                        let mut partial = Vec::with_capacity(group.len());
                        for (idx, item) in group {
                            if stop.as_ref().is_some_and(|f| f.load(Ordering::Relaxed)) {
                                break;
                            }
                            partial.push((idx, backend.predict(&item, &config, top_k)?));
                        }
                        Ok((partial, backend))
                    },
                ));
            }

            let mut returned_backends = Vec::with_capacity(handles.len());
            for handle in handles {
                // Check stop flag before waiting for each handle
                if stop_flag
                    .as_ref()
                    .is_some_and(|f| f.load(Ordering::Relaxed))
                {
                    // Don't wait for remaining handles; collect what we have.
                    // Note: spawned threads will still finish their current
                    // prediction but skip remaining items in their chunk.
                    break;
                }
                let (partial, backend) = handle
                    .join()
                    .map_err(|_| anyhow!("yolo worker thread panicked"))??;
                joined.extend(partial);
                returned_backends.push(backend);
            }
            Ok(returned_backends)
        })?;

        // Return backends to the pool for reuse by subsequent calls.
        // Recover from poisoned mutex if needed (clear stale data first).
        let mut pool = match self.backend_pool.lock() {
            Ok(guard) => guard,
            Err(e) => {
                let mut guard = e.into_inner();
                guard.clear();
                guard
            }
        };
        pool.extend(used_backends);

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
    class_names: Arc<[String]>,
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
                orig_img: Some(Arc::new(item.image.clone())),
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
        .resize_exact(resized_w, resized_h, imageops::FilterType::Triangle)
        .to_rgb8();
    let mut canvas = RgbImage::from_pixel(
        input_width as u32,
        input_height as u32,
        Rgb([114, 114, 114]),
    );
    let pad_x = ((input_width as u32).saturating_sub(resized_w)) / 2;
    let pad_y = ((input_height as u32).saturating_sub(resized_h)) / 2;
    imageops::replace(&mut canvas, &resized, pad_x as i64, pad_y as i64);
    let raw = canvas.as_raw();
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
            task,
            outputs,
            primary,
            meta,
            class_names,
            conf_threshold,
            iou_threshold,
            max_detections,
            top_k,
            nms_class_agnostic,
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
        YoloTaskKind::Classify => {
            decode_yolo_classify_output(primary, class_names, conf_threshold, top_k, max_detections)
        }
        YoloTaskKind::Segment => {
            let proto = find_mask_proto_output(outputs);
            if let Some(proto) = proto {
                try_decode_yolo_segment_output(
                    primary,
                    proto,
                    meta,
                    class_names,
                    conf_threshold,
                    iou_threshold,
                    max_detections,
                    nms_class_agnostic,
                )
                .map(|opt| {
                    opt.unwrap_or_else(|| DecodedYoloOutput {
                        task: YoloTaskKind::Segment,
                        ..Default::default()
                    })
                })
            } else {
                Ok(DecodedYoloOutput {
                    task: YoloTaskKind::Segment,
                    ..Default::default()
                })
            }
        }
        YoloTaskKind::Pose => try_decode_yolo_pose_output(
            primary,
            meta,
            class_names,
            conf_threshold,
            iou_threshold,
            max_detections,
            nms_class_agnostic,
            keypoint_confidence_threshold,
        )
        .map(|opt| {
            opt.unwrap_or_else(|| DecodedYoloOutput {
                task: YoloTaskKind::Pose,
                ..Default::default()
            })
        }),
        YoloTaskKind::Obb => try_decode_yolo_obb_output(
            primary,
            meta,
            class_names,
            conf_threshold,
            iou_threshold,
            max_detections,
            nms_class_agnostic,
        )
        .map(|opt| {
            opt.unwrap_or_else(|| DecodedYoloOutput {
                task: YoloTaskKind::Obb,
                ..Default::default()
            })
        }),
        YoloTaskKind::Detect => {
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
    let channel_candidates = infer_proto_channel_candidates(&proto.shape);
    if channel_candidates.is_empty() {
        return Ok(None);
    }
    let layout = prediction_layout(&tensor.shape, 6)?;

    // Try each candidate channel count until one produces a valid aux layout
    // and at least one detection. This handles ambiguity when dim1 == dim3
    // in the proto shape (e.g. [1, 32, 32, 32]).
    for mask_dim in channel_candidates {
        let Some(aux_layout) =
            infer_fixed_extra_layout(layout.attr_count, class_names.len(), mask_dim)
        else {
            continue;
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

        let selected = non_max_suppression_with_aux(
            candidates,
            iou_threshold,
            max_detections,
            nms_class_agnostic,
        );
        if selected.is_empty() {
            // This candidate produced no detections — still a valid segment layout,
            // just no confident predictions. Return empty segment result.
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
        return Ok(Some(DecodedYoloOutput {
            task: YoloTaskKind::Segment,
            boxes,
            masks,
            ..Default::default()
        }));
    }

    // No candidate channel count produced a valid layout
    Ok(None)
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
        if let Some(obb) =
            decode_obb_prediction(&prediction, meta, class_names, conf_threshold, aux_layout)
        {
            candidates.push(obb);
        }
    }

    candidates.sort_by(|left, right| right.conf.total_cmp(&left.conf));
    let obb = non_max_suppression_obb(
        candidates,
        iou_threshold,
        max_detections,
        nms_class_agnostic,
    );
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

    let selected = non_max_suppression_with_aux(
        candidates,
        iou_threshold,
        max_detections,
        nms_class_agnostic,
    );
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

fn select_primary_output<'a>(
    outputs: &'a [OnnxOutputTensor<'a>],
) -> Option<&'a OnnxOutputTensor<'a>> {
    // Priority: 3D tensor (standard detection/segment/pose/obb output) > 2D/1D
    // (classification). 4D tensors are mask protos and excluded here.
    outputs
        .iter()
        .find(|tensor| tensor.shape.len() == 3)
        .or_else(|| outputs.iter().find(|tensor| tensor.shape.len() != 4))
        .or_else(|| outputs.first())
}

fn find_mask_proto_output<'a>(
    outputs: &'a [OnnxOutputTensor<'a>],
) -> Option<&'a OnnxOutputTensor<'a>> {
    outputs.iter().find(|tensor| tensor.shape.len() == 4)
}

/// Maximum plausible class count for classification outputs.
/// Detection outputs typically have attribute counts in the range 6..=120,
/// while classification outputs have class counts in the range 2..=10000.
/// A value above this threshold is almost certainly not a classification output.
const MAX_CLASSIFICATION_CLASSES: i64 = 100_000;

fn is_classification_output(tensor: &OnnxOutputTensor<'_>) -> bool {
    match tensor.shape.as_slice() {
        // Standard classification shapes: [C], [1, C], [1, C, 1]
        // C must be > 1 (at least 2 classes) and within a plausible range.
        [classes] => *classes > 1 && *classes <= MAX_CLASSIFICATION_CLASSES,
        [1, classes] => *classes > 1 && *classes <= MAX_CLASSIFICATION_CLASSES,
        [1, classes, 1] => *classes > 1 && *classes <= MAX_CLASSIFICATION_CLASSES,
        // [1, 1, C] is ambiguous — could be a single-prediction detection output.
        // Only treat as classification if C is large enough that it cannot be
        // a detection attribute count (detection attrs are typically < 120).
        [1, 1, classes] => *classes > 120 && *classes <= MAX_CLASSIFICATION_CLASSES,
        _ => false,
    }
}

fn is_probable_obb_output(tensor: &OnnxOutputTensor<'_>) -> bool {
    let name = tensor.name.to_ascii_lowercase();
    name.contains("obb") || name.contains("angle") || name.contains("rot") || name.contains("xywhr")
}

/// Maximum plausible attribute count for YOLO detection outputs.
/// Detection attributes include 4 bbox coords + optional objectness + class scores
/// + possible extras (mask coeffs, keypoint dims, angle). Typical ranges:
/// - Detect: 4 + ~80 classes = ~84
/// - Segment: 4 + ~80 + 32 mask coeffs = ~116
/// - Pose: 4 + ~80 + 17×3 keypoints = ~135
/// - OBB: 4 + ~80 + 1 angle = ~85
/// 300 provides comfortable headroom for all known variants.
const MAX_PLAUSIBLE_ATTR_COUNT: usize = 300;

fn prediction_layout(shape: &[i64], min_attr_count: usize) -> Result<PredictionLayout> {
    if shape.len() != 3 {
        return Err(anyhow!("unsupported yolo output shape: {:?}", shape));
    }
    // Support batch dimension: if batch > 1 we still process as if batch=1
    // because our inference always uses batch_size=1. Dynamic batch models
    // may report batch as -1 or > 1 in the shape metadata, but the actual
    // output at runtime always has batch=1 for our single-image inference.
    let batch = shape[0];
    if batch != 1 && batch != -1 {
        // Log a warning but don't fail — the actual data has batch=1 at runtime.
        eprintln!(
            "[yolo] warning: output batch dimension is {} but expected 1; proceeding with batch=1",
            batch
        );
    }
    let dim1 = shape[1].max(1) as usize;
    let dim2 = shape[2].max(1) as usize;

    // Determine transposition from the output shape.
    //
    // YOLO outputs come in two layouts:
    //   - Normal (non-transposed): [1, num_predictions, num_attrs]
    //     e.g. YOLOv8 detect: [1, 8400, 84]
    //   - Transposed: [1, num_attrs, num_predictions]
    //     e.g. some YOLOv5 exports: [1, 85, 8400]
    //
    // The key insight is that attribute counts are small (6..~300) while
    // prediction counts are large (typically 1000+). We use this to disambiguate:
    //   - If dim2 is small (≤300) and dim1 is large (>300): normal layout
    //   - If dim1 is small (≤300) and dim2 is large (>300): transposed layout
    //   - Otherwise (both small or both large): default to normal layout
    let (pred_count, attr_count, transposed) = if dim2 >= min_attr_count
        && dim2 <= MAX_PLAUSIBLE_ATTR_COUNT
        && dim1 > MAX_PLAUSIBLE_ATTR_COUNT
    {
        // dim1 is large (predictions), dim2 is small (attrs) → normal [1, preds, attrs]
        (dim1, dim2, false)
    } else if dim1 >= min_attr_count
        && dim1 <= MAX_PLAUSIBLE_ATTR_COUNT
        && dim2 > MAX_PLAUSIBLE_ATTR_COUNT
    {
        // dim1 is small (attrs), dim2 is large (predictions) → transposed [1, attrs, preds]
        (dim2, dim1, true)
    } else if dim2 >= min_attr_count {
        // Ambiguous (both dimensions within plausible range, or both large).
        // Default to normal (non-transposed) layout, which is the standard
        // for YOLOv8+ and most ONNX exports.
        (dim1, dim2, false)
    } else if dim1 >= min_attr_count {
        // dim2 < min_attr_count, dim1 might be attrs → treat as transposed
        (dim2, dim1, true)
    } else {
        return Err(anyhow!(
            "unsupported yolo attribute count for shape {:?} (min_attr_count={})",
            shape,
            min_attr_count
        ));
    };

    if attr_count < min_attr_count {
        return Err(anyhow!(
            "unsupported yolo attribute count {attr_count} for shape {:?}",
            shape
        ));
    }
    Ok(PredictionLayout {
        pred_count,
        attr_count,
        transposed,
    })
}

fn prediction_at<'a>(
    values: &'a [f32],
    layout: &PredictionLayout,
    pred_index: usize,
) -> Cow<'a, [f32]> {
    if layout.transposed {
        Cow::Owned(
            (0..layout.attr_count)
                .map(|attr_index| values[attr_index * layout.pred_count + pred_index])
                .collect(),
        )
    } else {
        let start = pred_index * layout.attr_count;
        Cow::Borrowed(&values[start..start + layout.attr_count])
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
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
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

fn infer_pose_aux_layout(
    attr_count: usize,
    preferred_class_count: usize,
) -> Option<PredictionAuxLayout> {
    // Keypoints are stored as (x, y, confidence) triplets, so the minimum
    // extra dimension count for a valid pose output is 6 (2 keypoints × 3).
    const MIN_KEYPOINT_EXTRA_DIMS: usize = 6;
    for has_objectness in [true, false] {
        let base = if has_objectness { 5 } else { 4 };
        if attr_count <= base + preferred_class_count + MIN_KEYPOINT_EXTRA_DIMS.saturating_sub(1) {
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
    let detection =
        decode_aux_prediction(prediction, meta, class_names, conf_threshold, aux_layout)?;
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
    let detection =
        decode_aux_prediction(prediction, meta, class_names, conf_threshold, aux_layout)?;
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
    let detection =
        decode_aux_prediction(prediction, meta, class_names, conf_threshold, aux_layout)?;
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

/// Return candidate channel counts for a 4D proto tensor.
/// For NCHW `[1, C, H, W]` the channel count is dim1; for NHWC `[1, H, W, C]` it is dim3.
/// When both dims differ, we return both candidates ordered by likelihood (smaller first,
/// since channel counts are typically much smaller than spatial dimensions).
/// When they are equal, we return a single candidate.
fn infer_proto_channel_candidates(shape: &[i64]) -> Vec<usize> {
    if shape.len() != 4 {
        return Vec::new();
    }
    let c_first = shape[1].max(1) as usize;
    let c_last = shape[3].max(1) as usize;
    if c_first == c_last {
        vec![c_first]
    } else if c_first < c_last {
        // NCHW is more likely (small dim1 = channels), but try both
        vec![c_first, c_last]
    } else {
        // NHWC is more likely (small dim3 = channels), but try both
        vec![c_last, c_first]
    }
}

fn decode_rect_mask(meta: &LetterboxMeta, xyxy: &[f32; 4]) -> YoloMask {
    let width = meta.orig_w;
    let height = meta.orig_h;
    let mut data = vec![0_u8; width as usize * height as usize];
    let x1 = xyxy[0].floor().clamp(0.0, width as f32) as u32;
    let y1 = xyxy[1].floor().clamp(0.0, height as f32) as u32;
    let x2 = xyxy[2].ceil().clamp(0.0, width as f32) as u32;
    let y2 = xyxy[3].ceil().clamp(0.0, height as f32) as u32;
    let bbox = if x2 > x1 && y2 > y1 {
        (y1, y2, x1, x2)
    } else {
        (0, 0, 0, 0)
    };
    for y in y1..y2 {
        for x in x1..x2 {
            let index = (y * width + x) as usize;
            if let Some(cell) = data.get_mut(index) {
                *cell = 255;
            }
        }
    }
    YoloMask {
        width,
        height,
        data,
        bbox,
    }
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
        return Ok(YoloMask {
            width,
            height,
            data,
            bbox: (0, 0, 0, 0),
        });
    }

    let (channels, proto_h, proto_w, nchw) = {
        let dim1 = proto.shape[1].max(1) as usize;
        let dim2 = proto.shape[2].max(1) as usize;
        let dim3 = proto.shape[3].max(1) as usize;

        let nchw_matches = dim1 == coeffs.len();
        let nhwc_matches = dim3 == coeffs.len();

        if nchw_matches && nhwc_matches {
            // Ambiguous: dim1 == dim3 == coeffs.len(). Disambiguate by
            // convention — in NCHW the channel dimension is typically much
            // smaller than spatial dimensions (e.g. [1, 32, 160, 160]),
            // while in NHWC it's the last dimension (e.g. [1, 160, 160, 32]).
            // If dim2 > dim1, then NCHW is more likely (channel is small).
            // If dim2 < dim1, then NHWC is more likely (spatial is small).
            // If dim2 == dim1, default to NCHW (Ultralytics standard).
            let is_nchw = dim2 >= dim1;
            if is_nchw {
                (dim1, dim2, dim3, true)
            } else {
                (dim3, dim1, dim2, false)
            }
        } else if nchw_matches {
            (dim1, dim2, dim3, true)
        } else if nhwc_matches {
            (dim3, dim1, dim2, false)
        } else {
            return Err(anyhow!("proto channel count does not match coeffs"));
        }
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
            // Compute dot product of coeffs with proto column at (px, py).
            // For NCHW layout the channel values are contiguous at
            //   [c*proto_h*proto_w + py*proto_w + px] for each c,
            // so stride = proto_h * proto_w.
            // For NHWC layout they are contiguous at
            //   [py*proto_w*channels + px*channels + 0..channels],
            // which allows a direct slice dot-product.
            let logit = if nchw {
                let stride = proto_h * proto_w;
                let base = py * proto_w + px;
                coeffs
                    .iter()
                    .enumerate()
                    .map(|(c, &coeff)| {
                        coeff * proto.values.get(c * stride + base).copied().unwrap_or(0.0)
                    })
                    .sum::<f32>()
            } else {
                let base = py * proto_w * channels + px * channels;
                proto
                    .values
                    .get(base..base + channels)
                    .map(|slice| {
                        slice
                            .iter()
                            .zip(coeffs.iter())
                            .map(|(&p, &c)| c * p)
                            .sum::<f32>()
                    })
                    .unwrap_or(0.0)
            };
            if normalize_confidence(logit) >= 0.5 {
                let index = (y * width + x) as usize;
                if let Some(cell) = data.get_mut(index) {
                    *cell = 255;
                }
            }
        }
    }

    let bbox = compute_mask_bbox(&data, width, x1, y1, x2, y2);
    Ok(YoloMask {
        width,
        height,
        data,
        bbox,
    })
}

fn restore_point(x: f32, y: f32, meta: &LetterboxMeta) -> [f32; 2] {
    [
        ((x - meta.pad_x) / meta.scale).clamp(0.0, meta.orig_w as f32),
        ((y - meta.pad_y) / meta.scale).clamp(0.0, meta.orig_h as f32),
    ]
}

fn normalize_obb_angle(value: f32) -> f32 {
    // Ultralytics OBB export always outputs the angle as sigmoid(angle_raw) * π,
    // producing values in [0, π]. However, some export formats (e.g. full-precision
    // ONNX) may already apply sigmoid but forget the *π scaling, yielding a value
    // in [0, 1], while others output raw logits (large magnitude).
    //
    // Detection heuristic (order matters — check narrow ranges first):
    //   - If value is in [1, π], it's already a decoded angle in radians.
    //     Note: value=1.0 is treated as 1.0 radian (≈57.3°), not sigmoid*π.
    //     This is because a sigmoid output of exactly 1.0 requires logit→∞,
    //     which is extremely unlikely in practice, while 1.0 radian is a
    //     common angle value.
    //   - If value is in (0, 1), it's a sigmoid output missing the *π scaling.
    //   - If value is exactly 0 or very close, treat as angle 0 (not sigmoid).
    //   - Otherwise (large magnitude or negative), treat as raw logit and apply
    //     sigmoid * π.
    if value >= 1.0 && value <= std::f32::consts::PI {
        // Already a valid angle in radians (e.g. 1.0 rad ≈ 57.3°, 1.5 rad, etc.)
        value
    } else if value > 0.0 && value < 1.0 {
        // Sigmoid output missing the π scaling — apply it.
        // Values in (0, 1) from sigmoid(angle_raw) need *π to become radians.
        value * std::f32::consts::PI
    } else if value.abs() < 1e-6 {
        // Near-zero angle — could be either format, return 0 either way.
        0.0
    } else {
        // Raw logit or out-of-range — apply sigmoid then scale.
        normalize_confidence(value) * std::f32::consts::PI
    }
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

/// Apply sigmoid normalization if the value appears to be a raw logit.
///
/// YOLOv5/v7 outputs raw objectness logits (large magnitude), while YOLOv8/v9/v10
/// outputs sigmoid-activated class scores in [0, 1]. The heuristic distinguishes
/// between the two by checking whether the value is in a plausible probability range.
/// A small margin (0.05) is added above 1.0 to handle slight numerical overshoot
/// from sigmoid computations that may land just outside [0, 1].
///
/// # Heuristic details
///
/// - Values in (0.0, 1.05] are treated as pre-activated probabilities (YOLOv8 style).
/// - A value of exactly 0.0 is treated as a raw logit, because a sigmoid output of
///   0.0 would require a logit of -∞, which never occurs in practice. A raw logit
///   of 0.0 → sigmoid(0.0) = 0.5, which is the correct interpretation for YOLOv5
///   objectness scores.
/// - Negative values and values > 1.05 are treated as raw logits → sigmoid applied.
fn normalize_confidence(value: f32) -> f32 {
    if value > 0.0 && value <= 1.05 {
        value.min(1.0)
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
        // Apply normalize_confidence to handle both raw logits (YOLOv5 style)
        // and pre-activated probabilities (YOLOv8 style).
        let conf = normalize_confidence(prediction[4]);
        if conf < conf_threshold {
            return None;
        }
        // len==6 format: typically [x, y, w, h, conf, cls_id]
        // cls_id should be a non-negative integer (possibly with floating-point
        // noise from quantization, e.g. 2.9999 → 3). Validate that the value
        // is plausibly an integer class index before using it as such.
        // If the value looks like a probability (0.0..=1.0) rather than an
        // integer index, treat the prediction as single-class with that as
        // the confidence and re-check against the threshold.
        let cls_raw = prediction[5];
        let rounded = cls_raw.round();
        if rounded >= 0.0
            && (cls_raw - rounded).abs() < 0.1
            && (rounded as usize) < class_names.len().max(1)
        {
            // Looks like a valid class index (close to an integer, in range)
            let cls = (rounded as usize).min(class_names.len().saturating_sub(1));
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
        } else {
            // Not a valid class index — could be a single-class format where
            // element [5] is an additional score or probability. Treat as
            // class 0 with the existing confidence.
            let label = class_names
                .first()
                .cloned()
                .unwrap_or_else(|| "0".to_string());
            return Some(YoloBox {
                xywh: xyxy_to_xywh(xyxy),
                xyxy,
                conf,
                cls: 0,
                label,
            });
        }
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

/// Trait abstracting the NMS-relevant operations for a detection type.
trait NmsDetection: Clone {
    fn class_id(&self) -> usize;
    fn iou_with(&self, other: &Self) -> f32;
}

impl NmsDetection for YoloBox {
    fn class_id(&self) -> usize {
        self.cls
    }
    fn iou_with(&self, other: &Self) -> f32 {
        intersection_over_union(&self.xyxy, &other.xyxy)
    }
}

impl NmsDetection for YoloObb {
    fn class_id(&self) -> usize {
        self.cls
    }
    fn iou_with(&self, other: &Self) -> f32 {
        rotated_iou(&self.corners, &other.corners)
    }
}

impl<T: Clone> NmsDetection for (YoloBox, T) {
    fn class_id(&self) -> usize {
        self.0.cls
    }
    fn iou_with(&self, other: &Self) -> f32 {
        intersection_over_union(&self.0.xyxy, &other.0.xyxy)
    }
}

/// Generic Non-Maximum Suppression.
///
/// Assumes `detections` is already sorted by confidence in descending order.
/// When `nms_class_agnostic` is true, suppresses across all classes;
/// otherwise only suppresses within the same class.
fn nms<D: NmsDetection>(
    detections: Vec<D>,
    iou_threshold: f32,
    max_detections: usize,
    nms_class_agnostic: bool,
) -> Vec<D> {
    let mut selected = Vec::new();
    let mut suppressed = vec![false; detections.len()];
    for index in 0..detections.len() {
        if suppressed[index] || selected.len() >= max_detections {
            continue;
        }
        let current = &detections[index];
        for candidate_index in (index + 1)..detections.len() {
            if suppressed[candidate_index] {
                continue;
            }
            let same_class = detections[candidate_index].class_id() == current.class_id();
            if (nms_class_agnostic || same_class)
                && current.iou_with(&detections[candidate_index]) >= iou_threshold
            {
                suppressed[candidate_index] = true;
            }
        }
        selected.push(detections[index].clone());
    }
    selected
}

/// Convenience wrapper for standard detection NMS (boxes only).
fn non_max_suppression(
    detections: Vec<YoloBox>,
    iou_threshold: f32,
    max_detections: usize,
    nms_class_agnostic: bool,
) -> Vec<YoloBox> {
    nms(
        detections,
        iou_threshold,
        max_detections,
        nms_class_agnostic,
    )
}

/// Convenience wrapper for OBB NMS using rotated IoU.
fn non_max_suppression_obb(
    detections: Vec<YoloObb>,
    iou_threshold: f32,
    max_detections: usize,
    nms_class_agnostic: bool,
) -> Vec<YoloObb> {
    nms(
        detections,
        iou_threshold,
        max_detections,
        nms_class_agnostic,
    )
}

/// Convenience wrapper for NMS with auxiliary data (masks or keypoints).
fn non_max_suppression_with_aux<T: Clone>(
    detections: Vec<(YoloBox, T)>,
    iou_threshold: f32,
    max_detections: usize,
    nms_class_agnostic: bool,
) -> Vec<(YoloBox, T)> {
    nms(
        detections,
        iou_threshold,
        max_detections,
        nms_class_agnostic,
    )
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
    // Sutherland-Hodgman requires counter-clockwise winding for correct clipping.
    let ccw_a = ensure_counter_clockwise(corners_a);
    let ccw_b = ensure_counter_clockwise(corners_b);
    let area_a = polygon_area(&ccw_a);
    let area_b = polygon_area(&ccw_b);
    if area_a <= 0.0 || area_b <= 0.0 {
        return 0.0;
    }
    let inter = polygon_intersection_area(&ccw_a, &ccw_b);
    let union = area_a + area_b - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

/// Compute the signed area of a convex polygon using the shoelace formula.
/// A positive signed area indicates counter-clockwise winding; negative indicates clockwise.
fn signed_polygon_area(vertices: &[[f32; 2]; 4]) -> f32 {
    let n = vertices.len();
    let mut area = 0.0_f32;
    for i in 0..n {
        let j = (i + 1) % n;
        area += vertices[i][0] * vertices[j][1];
        area -= vertices[j][0] * vertices[i][1];
    }
    area * 0.5
}

/// Compute the unsigned area of a convex polygon using the shoelace formula.
fn polygon_area(vertices: &[[f32; 2]; 4]) -> f32 {
    signed_polygon_area(vertices).abs()
}

/// Ensure the polygon vertices are in counter-clockwise order by reversing
/// them if the signed area is negative (clockwise winding).
fn ensure_counter_clockwise(vertices: &[[f32; 2]; 4]) -> [[f32; 2]; 4] {
    if signed_polygon_area(vertices) < 0.0 {
        let mut v = *vertices;
        v.reverse();
        v
    } else {
        *vertices
    }
}

/// Compute the intersection area of two convex polygons using Sutherland-Hodgman clipping.
/// Clips polygon `subject` against each edge of polygon `clip`, then computes the
/// area of the resulting intersection polygon.
fn polygon_intersection_area(subject: &[[f32; 2]; 4], clip: &[[f32; 2]; 4]) -> f32 {
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
fn line_intersection(p1: [f32; 2], p2: [f32; 2], p3: [f32; 2], p4: [f32; 2]) -> Option<[f32; 2]> {
    let denom = (p1[0] - p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0] - p4[0]);
    if denom.abs() < 1e-10 {
        return None;
    }
    let t = ((p1[0] - p3[0]) * (p3[1] - p4[1]) - (p1[1] - p3[1]) * (p3[0] - p4[0])) / denom;
    Some([p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1])])
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

    // Heuristic: decide whether the prediction includes an objectness score
    // by examining whether the values after the 4 bbox coords look like raw
    // logits (large magnitude → objectness present, YOLOv5 style) or
    // probabilities (small magnitude → no objectness, YOLOv8 style).
    //
    // We use a two-step heuristic:
    // 1. Check if the majority of values after bbox are outside [−0.1, 1.1].
    //    If most values are raw logits, the prediction likely has objectness.
    //    Using majority vote avoids misclassification from a single value
    //    that slightly exceeds the sigmoid output range due to floating-point
    //    imprecision.
    // 2. If the counts are ambiguous, check the candidate objectness value
    //    itself: a valid objectness probability should be in [0, 1.05],
    //    while a raw logit is typically outside this range.
    let after_bbox = &prediction[4..];
    if after_bbox.is_empty() {
        return None;
    }

    let out_of_prob_range = |v: f32| v > 1.1 || v < -0.1;
    let out_count = after_bbox.iter().filter(|&&v| out_of_prob_range(v)).count();
    let has_objectness = if out_count > after_bbox.len() / 2 {
        // Majority of values are clearly logits → has objectness
        true
    } else if out_count == 0 {
        // All values in probability range → no objectness
        false
    } else {
        // Ambiguous: check whether the first value (candidate objectness)
        // looks like a raw logit or a probability.
        out_of_prob_range(after_bbox[0])
    };

    if has_objectness {
        // First value after bbox is objectness, rest are class scores (raw logits).
        Some((after_bbox[0], &after_bbox[1..]))
    } else {
        // No objectness — all values after bbox are class scores (already in [0,1]).
        Some((1.0, after_bbox))
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
            // Non-ASCII characters: draw a hollow 8x8 box as placeholder
            // (more distinguishable than a solid block)
            for row in 0..8u32 {
                let yy = y.saturating_add(row);
                if yy >= height {
                    break;
                }
                for col in 0..8u32 {
                    // Only draw the border pixels (hollow box), skip interior
                    if row > 0 && row < 7 && col > 0 && col < 7 {
                        continue;
                    }
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
            let path = match entry {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("[yolo] warning: glob entry failed: {e}, skipping");
                    continue;
                }
            };
            match resolve_local_path(&path, options) {
                Ok(resolved) => sources.extend(resolved),
                Err(e) => {
                    // Skip unsupported files (e.g. non-image/non-video)
                    // rather than failing the entire glob expansion.
                    eprintln!("[yolo] warning: skipping {}: {e}", path.display());
                }
            }
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
    source.starts_with("rtsp://") || source.starts_with("rtmp://") || source.starts_with("tcp://")
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
            &[
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string(),
            ],
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
            16.0_f32, 16.0, 8.0, 4.0, 0.95, 0.3, 8.0_f32, 8.0, 2.0, 2.0, 0.10, 0.1,
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
        // A classification-shaped output, but we force Detect mode.
        // Provide enough attributes for a valid detection layout (>=6).
        let predictions = vec![16.0_f32, 16.0, 8.0, 4.0, 0.9, 0.5];
        let output = OnnxOutputTensor {
            name: "output0".to_string(),
            shape: vec![1, 1, 6],
            values: &predictions,
        };
        let decoded = decode_yolo_outputs(
            &[output],
            &dummy_meta(),
            &["a".to_string()],
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
        let result_aware = non_max_suppression(vec![box1.clone(), box2.clone()], 0.5, 100, false);
        assert_eq!(result_aware.len(), 2);
        // Class-agnostic: second is suppressed
        let result_agnostic = non_max_suppression(vec![box1, box2], 0.5, 100, true);
        assert_eq!(result_agnostic.len(), 1);
    }

    #[test]
    fn keypoint_confidence_threshold_filters_low_confidence() {
        // 1 prediction with 1 class (no objectness), 2 keypoints (6 extra dims)
        // Layout: [cx, cy, w, h, class_0, kp1_x, kp1_y, kp1_conf, kp2_x, kp2_y, kp2_conf]
        //         4 base + 1 class + 6 kpts = 11 attributes
        let predictions: Vec<f32> = vec![
            16.0, 16.0, 8.0, 4.0, // cx, cy, w, h
            0.9, // class score (1 class)
            // 2 keypoints: x, y, conf
            10.0, 10.0, 0.8, 20.0, 20.0, 0.05,
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
        let corners = [[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]];
        let iou = rotated_iou(&corners, &corners);
        assert!(
            (iou - 1.0).abs() < 1e-4,
            "identical boxes should have IoU=1.0, got {iou}"
        );
    }

    #[test]
    fn rotated_iou_non_overlapping() {
        let corners_a = [[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]];
        let corners_b = [[20.0, 0.0], [30.0, 0.0], [30.0, 5.0], [20.0, 5.0]];
        let iou = rotated_iou(&corners_a, &corners_b);
        assert!(
            iou < 0.01,
            "non-overlapping boxes should have IoU≈0.0, got {iou}"
        );
    }

    #[test]
    fn rotated_iou_partial_overlap() {
        let corners_a = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        let corners_b = [[5.0, 0.0], [15.0, 0.0], [15.0, 10.0], [5.0, 10.0]];
        let iou = rotated_iou(&corners_a, &corners_b);
        // Intersection = 5*10 = 50, union = 100+100-50 = 150, IoU = 50/150 ≈ 0.333
        assert!(
            (iou - 0.333).abs() < 0.02,
            "partial overlap should have IoU≈0.333, got {iou}"
        );
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
        let result = non_max_suppression_obb(vec![obb_a, obb_b], 0.5, 100, false);
        assert_eq!(
            result.len(),
            2,
            "rotated boxes at different angles should both survive NMS"
        );
    }

    #[test]
    fn normalize_confidence_sigmoid_range() {
        // Values in (0, 1] should pass through unchanged (clamped to 1.0)
        assert!((normalize_confidence(0.5) - 0.5).abs() < 1e-6);
        assert!((normalize_confidence(1.0) - 1.0).abs() < 1e-6);
        // Very small positive value treated as probability
        assert!((normalize_confidence(0.001) - 0.001).abs() < 1e-6);
    }

    #[test]
    fn normalize_confidence_slight_overshoot_clamped() {
        // Values slightly above 1.0 should be clamped to 1.0
        assert!((normalize_confidence(1.04) - 1.0).abs() < 1e-6);
        assert!((normalize_confidence(1.05) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn normalize_confidence_raw_logit_applies_sigmoid() {
        // Large positive logit → sigmoid ≈ 1.0
        assert!(normalize_confidence(10.0) > 0.99);
        // Large negative logit → sigmoid ≈ 0.0
        assert!(normalize_confidence(-10.0) < 0.01);
        // Zero logit → sigmoid(0.0) = 0.5 (not treated as probability 0.0)
        assert!((normalize_confidence(0.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn normalize_confidence_negative_value() {
        // Negative values outside (0, 1.05] → sigmoid
        assert!((normalize_confidence(-5.0) - 1.0 / (1.0 + 5.0_f32.exp())).abs() < 1e-6);
    }

    #[test]
    fn normalize_obb_angle_radians_range() {
        // Values in [1, π] should pass through as angle in radians
        assert!((normalize_obb_angle(1.0) - 1.0).abs() < 1e-6);
        assert!((normalize_obb_angle(std::f32::consts::PI) - std::f32::consts::PI).abs() < 1e-6);
        assert!((normalize_obb_angle(1.5) - 1.5).abs() < 1e-6);
    }

    #[test]
    fn normalize_obb_angle_sigmoid_missing_pi() {
        // Values in (0, 1) should be multiplied by π
        assert!((normalize_obb_angle(0.5) - 0.5 * std::f32::consts::PI).abs() < 1e-6);
        assert!((normalize_obb_angle(0.1) - 0.1 * std::f32::consts::PI).abs() < 1e-6);
        assert!((normalize_obb_angle(0.99) - 0.99 * std::f32::consts::PI).abs() < 1e-4);
    }

    #[test]
    fn normalize_obb_angle_near_zero() {
        // Near-zero values should return 0
        assert!(normalize_obb_angle(0.0).abs() < 1e-6);
        assert!(normalize_obb_angle(1e-7).abs() < 1e-6);
        assert!(normalize_obb_angle(-1e-7).abs() < 1e-6);
    }

    #[test]
    fn normalize_obb_angle_raw_logit() {
        // Large magnitude values should be treated as raw logits → sigmoid * π
        let raw = 5.0_f32;
        let expected = normalize_confidence(raw) * std::f32::consts::PI;
        assert!((normalize_obb_angle(raw) - expected).abs() < 1e-6);

        // Negative large values
        let raw_neg = -5.0_f32;
        let expected_neg = normalize_confidence(raw_neg) * std::f32::consts::PI;
        assert!((normalize_obb_angle(raw_neg) - expected_neg).abs() < 1e-6);
    }

    #[test]
    fn polygon_intersection_fully_contained() {
        // A small box fully inside a larger box → intersection = small box area
        let big = [[0.0_f32, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0]];
        let small = [[5.0_f32, 5.0], [10.0, 5.0], [10.0, 10.0], [5.0, 10.0]];
        let area = polygon_intersection_area(&big, &small);
        assert!(
            (area - 25.0).abs() < 0.5,
            "fully contained intersection should be 25.0, got {area}"
        );
    }

    #[test]
    fn polygon_intersection_no_overlap() {
        let a = [[0.0_f32, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 5.0]];
        let b = [[10.0_f32, 10.0], [15.0, 10.0], [15.0, 15.0], [10.0, 15.0]];
        let area = polygon_intersection_area(&a, &b);
        assert!(
            area < 0.01,
            "non-overlapping polygons should have 0 intersection, got {area}"
        );
    }

    #[test]
    fn polygon_intersection_touching_edge() {
        // Two boxes sharing an edge but no interior overlap
        let a = [[0.0_f32, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        let b = [[10.0_f32, 0.0], [20.0, 0.0], [20.0, 10.0], [10.0, 10.0]];
        let area = polygon_intersection_area(&a, &b);
        assert!(
            area < 1.0,
            "edge-touching polygons should have near-zero intersection, got {area}"
        );
    }

    #[test]
    fn polygon_intersection_identical() {
        let square = [[0.0_f32, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        let area = polygon_intersection_area(&square, &square);
        assert!(
            (area - 100.0).abs() < 1.0,
            "identical polygons should have full intersection, got {area}"
        );
    }

    #[test]
    fn polygon_intersection_rotated_45() {
        // A square and the same square rotated 45° at the same center.
        // Intersection is the regular octagon inscribed in the square.
        let axis = oriented_box_corners([10.0, 10.0], 10.0, 10.0, 0.0);
        let rotated = oriented_box_corners([10.0, 10.0], 10.0, 10.0, std::f32::consts::FRAC_PI_4);
        let area = polygon_intersection_area(
            &ensure_counter_clockwise(&axis),
            &ensure_counter_clockwise(&rotated),
        );
        // The octagon area = 2 * (s^2) * (sqrt(2) - 1) where s = side/2 = 5
        // = 2 * 25 * 0.4142 ≈ 20.71 ... but with floating point, approximate
        assert!(
            area > 15.0,
            "rotated square intersection should be > 15, got {area}"
        );
        assert!(
            area < 100.0,
            "rotated square intersection should be < 100, got {area}"
        );
    }

    #[test]
    fn split_detection_prediction_majority_logit_detected() {
        // 4 bbox + objectness + 3 class logits (all raw, > 1.1)
        let prediction: Vec<f32> = vec![
            10.0, 10.0, 5.0, 5.0, // bbox
            3.5, // objectness (raw logit)
            2.0, 1.5, 0.5, // class scores (raw logits, majority > 1.1)
        ];
        let result = split_detection_prediction(&prediction, 0);
        assert!(result.is_some());
        let (obj, class_scores) = result.unwrap();
        assert!(
            (obj - 3.5).abs() < 1e-6,
            "objectness should be 3.5, got {obj}"
        );
        assert_eq!(class_scores.len(), 3, "should have 3 class scores");
    }

    #[test]
    fn split_detection_prediction_all_probabilities_no_objectness() {
        // 4 bbox + 3 class probabilities (all in [0, 1])
        let prediction: Vec<f32> = vec![
            10.0, 10.0, 5.0, 5.0, // bbox
            0.9, 0.05, 0.03, // class probabilities
        ];
        let result = split_detection_prediction(&prediction, 0);
        assert!(result.is_some());
        let (obj, class_scores) = result.unwrap();
        assert!(
            (obj - 1.0).abs() < 1e-6,
            "objectness should be 1.0 (no objectness), got {obj}"
        );
        assert_eq!(class_scores.len(), 3, "should have 3 class scores");
    }

    #[test]
    fn split_detection_prediction_single_slight_overshoot_ignored() {
        // 4 bbox + 3 class probabilities where one value slightly exceeds 1.05
        // but majority are in probability range → no objectness
        let prediction: Vec<f32> = vec![
            10.0, 10.0, 5.0, 5.0, // bbox
            0.9, 0.05, 1.07, // one value slightly > 1.05, but minority
        ];
        let result = split_detection_prediction(&prediction, 0);
        assert!(result.is_some());
        let (obj, class_scores) = result.unwrap();
        assert!(
            (obj - 1.0).abs() < 1e-6,
            "should detect no objectness, got obj={obj}"
        );
        assert_eq!(class_scores.len(), 3, "should have 3 class scores");
    }

    #[test]
    fn is_classification_output_rejects_small_1_1_n() {
        // [1, 1, 80] should NOT be classified as classification output
        // (80 attrs could be a small detection model)
        let tensor = OnnxOutputTensor {
            name: "output0".to_string(),
            shape: vec![1, 1, 80],
            values: &[0.1; 80],
        };
        assert!(
            !is_classification_output(&tensor),
            "[1, 1, 80] should not be treated as classification output"
        );
    }

    #[test]
    fn is_classification_output_accepts_large_1_1_n() {
        // [1, 1, 1000] should be classified as classification output
        let tensor = OnnxOutputTensor {
            name: "output0".to_string(),
            shape: vec![1, 1, 1000],
            values: &[0.1; 1000],
        };
        assert!(
            is_classification_output(&tensor),
            "[1, 1, 1000] should be treated as classification output"
        );
    }

    #[test]
    fn is_classification_output_accepts_standard_shapes() {
        let shapes: Vec<Vec<i64>> = vec![vec![1000], vec![1, 1000], vec![1, 1000, 1]];
        for shape in shapes {
            let tensor = OnnxOutputTensor {
                name: "output0".to_string(),
                shape,
                values: &[0.1; 10], // dummy
            };
            assert!(
                is_classification_output(&tensor),
                "shape {:?} should be classification",
                tensor.shape
            );
        }
    }

    #[test]
    fn select_primary_output_prefers_3d() {
        let values: Vec<f32> = vec![0.0; 10];
        let tensor_2d = OnnxOutputTensor {
            name: "output0".to_string(),
            shape: vec![1, 100],
            values: &values,
        };
        let tensor_3d = OnnxOutputTensor {
            name: "output1".to_string(),
            shape: vec![1, 8400, 6],
            values: &values,
        };
        let tensor_4d = OnnxOutputTensor {
            name: "proto".to_string(),
            shape: vec![1, 32, 160, 160],
            values: &values,
        };
        let outputs = [tensor_4d, tensor_2d, tensor_3d];
        let primary = select_primary_output(&outputs);
        assert!(primary.is_some());
        assert_eq!(
            primary.unwrap().shape.len(),
            3,
            "should select 3D tensor over 2D and 4D"
        );
    }

    #[test]
    fn prediction_layout_supports_dynamic_batch() {
        // batch = -1 (dynamic) should be accepted instead of returning an error
        let layout = prediction_layout(&[-1, 8400, 6], 6);
        assert!(layout.is_ok(), "dynamic batch shape should be accepted");

        // batch = 2 should also be accepted with a warning instead of an error
        let layout2 = prediction_layout(&[2, 8400, 6], 6);
        assert!(layout2.is_ok(), "batch=2 shape should be accepted");
    }

    #[test]
    fn prediction_layout_yolov8_normal_not_transposed() {
        // YOLOv8 standard output [1, 8400, 84] should be detected as normal (non-transposed)
        let layout = prediction_layout(&[1, 8400, 84], 6).unwrap();
        assert_eq!(
            layout.pred_count, 8400,
            "YOLOv8 [1,8400,84] pred_count should be 8400"
        );
        assert_eq!(
            layout.attr_count, 84,
            "YOLOv8 [1,8400,84] attr_count should be 84"
        );
        assert!(
            !layout.transposed,
            "YOLOv8 [1,8400,84] should NOT be transposed"
        );
    }

    #[test]
    fn prediction_layout_yolov5_transposed() {
        // YOLOv5 transposed output [1, 85, 8400] should be detected as transposed
        let layout = prediction_layout(&[1, 85, 8400], 6).unwrap();
        assert_eq!(
            layout.pred_count, 8400,
            "YOLOv5 [1,85,8400] pred_count should be 8400"
        );
        assert_eq!(
            layout.attr_count, 85,
            "YOLOv5 [1,85,8400] attr_count should be 85"
        );
        assert!(layout.transposed, "YOLOv5 [1,85,8400] should be transposed");
    }

    #[test]
    fn prediction_layout_yolov8_segment_normal() {
        // YOLOv8 segment output [1, 8400, 116] should be normal
        let layout = prediction_layout(&[1, 8400, 116], 6).unwrap();
        assert_eq!(layout.pred_count, 8400);
        assert_eq!(layout.attr_count, 116);
        assert!(!layout.transposed);
    }

    #[test]
    fn prediction_layout_yolov5_non_transposed() {
        // YOLOv5 non-transposed output [1, 25200, 85] should be normal
        let layout = prediction_layout(&[1, 25200, 85], 6).unwrap();
        assert_eq!(layout.pred_count, 25200);
        assert_eq!(layout.attr_count, 85);
        assert!(!layout.transposed);
    }

    #[test]
    fn prediction_layout_small_ambiguous_defaults_normal() {
        // Small ambiguous shape [1, 20, 6] — both dims within plausible range,
        // should default to normal (non-transposed)
        let layout = prediction_layout(&[1, 20, 6], 6).unwrap();
        assert_eq!(layout.pred_count, 20);
        assert_eq!(layout.attr_count, 6);
        assert!(!layout.transposed);
    }

    #[test]
    fn prediction_layout_yolov8_obb_normal() {
        // YOLOv8 OBB output [1, 8400, 85] should be normal
        let layout = prediction_layout(&[1, 8400, 85], 6).unwrap();
        assert_eq!(layout.pred_count, 8400);
        assert_eq!(layout.attr_count, 85);
        assert!(!layout.transposed);
    }

    #[test]
    fn prediction_layout_yolov8_pose_normal() {
        // YOLOv8 pose output [1, 8400, 56] should be normal
        let layout = prediction_layout(&[1, 8400, 56], 6).unwrap();
        assert_eq!(layout.pred_count, 8400);
        assert_eq!(layout.attr_count, 56);
        assert!(!layout.transposed);
    }

    #[test]
    fn decode_single_prediction_len6_valid_cls_id() {
        // len==6 with valid integer class ID
        let prediction: Vec<f32> = vec![16.0, 16.0, 8.0, 4.0, 0.9, 2.0];
        let result = decode_single_prediction(
            &prediction,
            &dummy_meta(),
            &["a".to_string(), "b".to_string(), "c".to_string()],
            0.5,
        );
        assert!(result.is_some());
        let box_result = result.unwrap();
        assert_eq!(box_result.cls, 2);
        assert_eq!(box_result.label, "c");
    }

    #[test]
    fn decode_single_prediction_len6_quantized_cls_id() {
        // len==6 with quantized class ID (2.9999 → 3)
        let prediction: Vec<f32> = vec![16.0, 16.0, 8.0, 4.0, 0.9, 2.9999];
        let result = decode_single_prediction(
            &prediction,
            &dummy_meta(),
            &[
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string(),
            ],
            0.5,
        );
        assert!(result.is_some());
        let box_result = result.unwrap();
        assert_eq!(box_result.cls, 3);
        assert_eq!(box_result.label, "d");
    }

    #[test]
    fn decode_single_prediction_len6_non_integer_falls_back_to_class0() {
        // len==6 where prediction[5] is clearly not an integer class ID (0.85)
        let prediction: Vec<f32> = vec![16.0, 16.0, 8.0, 4.0, 0.9, 0.85];
        let result = decode_single_prediction(
            &prediction,
            &dummy_meta(),
            &["cat".to_string(), "dog".to_string()],
            0.5,
        );
        assert!(result.is_some());
        let box_result = result.unwrap();
        assert_eq!(
            box_result.cls, 0,
            "non-integer cls value should fall back to class 0"
        );
        assert_eq!(box_result.label, "cat");
    }

    #[test]
    fn infer_proto_channel_candidates_nchw() {
        // Standard NCHW proto: [1, 32, 160, 160] — dim1=32 ≠ dim3=160
        // Should return both candidates, smaller first (NCHW more likely)
        let candidates = infer_proto_channel_candidates(&[1, 32, 160, 160]);
        assert_eq!(
            candidates,
            vec![32, 160],
            "NCHW with distinct dims should return both candidates"
        );
        assert_eq!(
            candidates[0], 32,
            "first candidate should be 32 (NCHW channel dim)"
        );
    }

    #[test]
    fn infer_proto_channel_candidates_nhwc() {
        // NHWC proto: [1, 160, 160, 32]
        let candidates = infer_proto_channel_candidates(&[1, 160, 160, 32]);
        assert!(!candidates.is_empty());
        assert_eq!(
            candidates[0], 32,
            "first candidate should be 32 (smaller dim)"
        );
    }

    #[test]
    fn infer_proto_channel_candidates_ambiguous() {
        // Ambiguous: [1, 32, 32, 32] — both dims are 32
        let candidates = infer_proto_channel_candidates(&[1, 32, 32, 32]);
        assert_eq!(
            candidates,
            vec![32],
            "equal dims should produce single candidate"
        );
    }
}
