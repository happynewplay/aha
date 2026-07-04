use std::sync::{Arc, OnceLock};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default, clap::ValueEnum)]
pub enum YoloTaskKind {
    #[default]
    Detect,
    Segment,
    Pose,
    Classify,
    Obb,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YoloConfig {
    pub image_size: usize,
    pub confidence_threshold: f32,
    pub iou_threshold: f32,
    pub max_detections: usize,
    #[serde(
        serialize_with = "serialize_arc_slice",
        deserialize_with = "deserialize_arc_slice"
    )]
    pub class_names: Arc<[String]>,
    /// If set, overrides the auto-detected task kind from ONNX output.
    pub task_kind: Option<YoloTaskKind>,
    /// When true, NMS suppresses overlapping boxes across all classes (class-agnostic).
    /// When false (default), NMS only suppresses within the same class.
    pub nms_class_agnostic: bool,
    /// Keypoints with confidence below this threshold are excluded from results.
    pub keypoint_confidence_threshold: f32,
    /// When true, keep the original image in YoloResults.orig_img for plotting.
    /// When false, orig_img is discarded to save memory (default: true).
    pub keep_images: bool,
}

pub fn serialize_arc_slice<S: serde::Serializer>(
    val: &Arc<[String]>,
    s: S,
) -> Result<S::Ok, S::Error> {
    s.collect_seq(val.iter())
}

pub fn deserialize_arc_slice<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> Result<Arc<[String]>, D::Error> {
    let vec: Vec<String> = Vec::deserialize(d)?;
    Ok(vec.into())
}

impl Default for YoloConfig {
    fn default() -> Self {
        Self {
            image_size: 640,
            confidence_threshold: 0.25,
            iou_threshold: 0.45,
            max_detections: 300,
            class_names: default_coco_class_names(),
            task_kind: None,
            nms_class_agnostic: false,
            keypoint_confidence_threshold: 0.1,
            keep_images: true,
        }
    }
}

impl YoloConfig {
    /// Validate the configuration and return a normalized version.
    ///
    /// Clamps out-of-range values to their valid bounds and logs warnings
    /// for values that were adjusted. This is called automatically by
    /// `YoloModel::init_with_config`, but can also be called manually.
    pub fn validate(self) -> Self {
        let mut config = self;
        if config.confidence_threshold < 0.0 {
            eprintln!(
                "[yolo] warning: confidence_threshold={} is negative, clamping to 0.0",
                config.confidence_threshold
            );
            config.confidence_threshold = 0.0;
        }
        if config.confidence_threshold > 1.0 {
            eprintln!(
                "[yolo] warning: confidence_threshold={} > 1.0, clamping to 1.0",
                config.confidence_threshold
            );
            config.confidence_threshold = 1.0;
        }
        if config.iou_threshold < 0.0 {
            eprintln!(
                "[yolo] warning: iou_threshold={} is negative, clamping to 0.0",
                config.iou_threshold
            );
            config.iou_threshold = 0.0;
        }
        if config.iou_threshold > 1.0 {
            eprintln!(
                "[yolo] warning: iou_threshold={} > 1.0, clamping to 1.0",
                config.iou_threshold
            );
            config.iou_threshold = 1.0;
        }
        if config.max_detections == 0 {
            eprintln!("[yolo] warning: max_detections=0 makes no sense, setting to 1");
            config.max_detections = 1;
        }
        if config.keypoint_confidence_threshold < 0.0 {
            eprintln!(
                "[yolo] warning: keypoint_confidence_threshold={} is negative, clamping to 0.0",
                config.keypoint_confidence_threshold
            );
            config.keypoint_confidence_threshold = 0.0;
        }
        if config.keypoint_confidence_threshold > 1.0 {
            eprintln!(
                "[yolo] warning: keypoint_confidence_threshold={} > 1.0, clamping to 1.0",
                config.keypoint_confidence_threshold
            );
            config.keypoint_confidence_threshold = 1.0;
        }
        if config.image_size == 0 {
            eprintln!("[yolo] warning: image_size=0 is invalid, setting to 640");
            config.image_size = 640;
        }
        config
    }
}

pub fn default_coco_class_names() -> Arc<[String]> {
    static CACHE: OnceLock<Arc<[String]>> = OnceLock::new();
    CACHE
        .get_or_init(|| {
            [
                "person",
                "bicycle",
                "car",
                "motorcycle",
                "airplane",
                "bus",
                "train",
                "truck",
                "boat",
                "traffic light",
                "fire hydrant",
                "stop sign",
                "parking meter",
                "bench",
                "bird",
                "cat",
                "dog",
                "horse",
                "sheep",
                "cow",
                "elephant",
                "bear",
                "zebra",
                "giraffe",
                "backpack",
                "umbrella",
                "handbag",
                "tie",
                "suitcase",
                "frisbee",
                "skis",
                "snowboard",
                "sports ball",
                "kite",
                "baseball bat",
                "baseball glove",
                "skateboard",
                "surfboard",
                "tennis racket",
                "bottle",
                "wine glass",
                "cup",
                "fork",
                "knife",
                "spoon",
                "bowl",
                "banana",
                "apple",
                "sandwich",
                "orange",
                "broccoli",
                "carrot",
                "hot dog",
                "pizza",
                "donut",
                "cake",
                "chair",
                "couch",
                "potted plant",
                "bed",
                "dining table",
                "toilet",
                "tv",
                "laptop",
                "mouse",
                "remote",
                "keyboard",
                "cell phone",
                "microwave",
                "oven",
                "toaster",
                "sink",
                "refrigerator",
                "book",
                "clock",
                "vase",
                "scissors",
                "teddy bear",
                "hair drier",
                "toothbrush",
            ]
            .into_iter()
            .map(str::to_string)
            .collect()
        })
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_clamps_negative_confidence() {
        let config = YoloConfig {
            confidence_threshold: -0.5,
            ..Default::default()
        };
        let validated = config.validate();
        assert!((validated.confidence_threshold - 0.0).abs() < 1e-6);
    }

    #[test]
    fn validate_clamps_confidence_above_one() {
        let config = YoloConfig {
            confidence_threshold: 2.0,
            ..Default::default()
        };
        let validated = config.validate();
        assert!((validated.confidence_threshold - 1.0).abs() < 1e-6);
    }

    #[test]
    fn validate_clamps_iou_out_of_range() {
        let config = YoloConfig {
            iou_threshold: -0.1,
            ..Default::default()
        };
        let validated = config.validate();
        assert!((validated.iou_threshold - 0.0).abs() < 1e-6);

        let config = YoloConfig {
            iou_threshold: 1.5,
            ..Default::default()
        };
        let validated = config.validate();
        assert!((validated.iou_threshold - 1.0).abs() < 1e-6);
    }

    #[test]
    fn validate_fixes_zero_max_detections() {
        let config = YoloConfig {
            max_detections: 0,
            ..Default::default()
        };
        let validated = config.validate();
        assert_eq!(validated.max_detections, 1);
    }

    #[test]
    fn validate_fixes_zero_image_size() {
        let config = YoloConfig {
            image_size: 0,
            ..Default::default()
        };
        let validated = config.validate();
        assert_eq!(validated.image_size, 640);
    }

    #[test]
    fn validate_passes_valid_config_unchanged() {
        let config = YoloConfig::default();
        let validated = config.clone().validate();
        assert!((validated.confidence_threshold - config.confidence_threshold).abs() < 1e-6);
        assert!((validated.iou_threshold - config.iou_threshold).abs() < 1e-6);
        assert_eq!(validated.max_detections, config.max_detections);
        assert_eq!(validated.image_size, config.image_size);
    }
}
