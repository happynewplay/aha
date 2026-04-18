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
    pub class_names: Vec<String>,
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

pub fn default_coco_class_names() -> Vec<String> {
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
}
