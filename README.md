# Enhanced Billboard & Signboard Detection (Zero‑Shot with OWLv2 + CV Ensemble)

This project detects billboards and signboards in images without any labeled training data. It uses a zero‑shot, open‑vocabulary detector (OWLv2) and combines it with classic computer vision (edges + contours) in an ensemble to improve robustness. It is designed to run efficiently on a GPU (optimized for A100 with FP16), but also works on CPU with reduced speed.

- Zero‑shot detector: OWLv2 Large (Patch14) via Hugging Face transformers
- CV fallback: Canny edges + contour analysis + geometry/text‑like heuristics
- Ensemble: Weighted combination (open‑vocab 0.8, CV 0.2) with per‑class NMS and size/aspect filters
- Outputs: 
  - Saved images containing billboards to one folder
  - Saved images containing signboards to another folder
  - Detailed CSV of per‑image results

---

## Why open‑vocabulary detection (and not YOLO‑World / Grounding DINO / others)?

- No labeled dataset available. Supervised detectors like standard YOLO variants typically need labeled bounding boxes for (re)training to reach strong performance on a specific domain (billboards/signboards in the wild).  
- Zero‑shot requirement. OWLv2 supports text prompts (e.g., “large advertising billboard”, “shop signboard”) and can generalize to unseen categories without finetuning.
- Setup friction vs. time. Grounding DINO and YOLO‑World are strong open‑vocabulary options but often require:
  - Heavier/complex dependencies and CUDA/TensorRT builds (especially for best performance).
  - Different text encoders/tokenizers and/or extra steps to get stable, reproducible pipelines.
  - More tuning to integrate cleanly with a CPU/GPU fall‑back and an ensemble with classic CV features.

This code prioritizes a single‑file, reproducible, zero‑shot pipeline. With no labeled data, OWLv2 provides a pragmatic balance of accuracy, setup simplicity, and inference speed.

---

## What this code does

1. OWLv2 open‑vocabulary detection
   - Queries two concept sets independently:
     - Billboard queries (e.g., “roadside billboard”, “digital billboard”)
     - Signboard queries (e.g., “shop signboard”, “store sign”, “road sign”)
   - Post‑filters by geometry and size:
     - Billboard: larger, wider rectangles (min area/width/height ratios; aspect ratio range)
     - Signboard: smaller/more flexible shapes
   - Applies per‑category NMS (IoU 0.4)
   - Runs in FP16 on CUDA for speed when available

2. CV heuristic detection
   - Canny edges + contour approximation to find rectangular-ish regions
   - Size/aspect checks to roughly classify billboard vs. signboard
   - Morphological close to estimate “text‑like” regions for additional confidence

3. Ensemble decision
   - Weighted confidence: open‑vocab (0.8) + CV (0.2)
   - Declare detection if open‑vocab is positive, or CV is positive and ensemble score > 0.6

4. Batch processing
   - Multi-threaded image processing with progress bar
   - Saves positives to:
     - billboards: /content/drive/MyDrive/billboard_detected
     - signboards: /content/drive/MyDrive/signboard_detected
   - Writes a CSV with detailed per‑image metrics

---

## About the model used: OWLv2

OWLv2 (Open‑World Localization) is a text‑conditioned object detector that can detect objects given natural language queries, without finetuning on labeled bounding boxes for those specific classes.

- Checkpoints commonly used:
  - google/owlv2-base-patch16
  - google/owlv2-large-patch14
  - google/owlv2-large-patch14-ensemble  ← used here for better recall
- “Large” vs. “Base”: Larger models typically provide better zero‑shot recall/precision but require more compute and memory.
- “Ensemble” variant: Combines multi‑resolution / multi‑head predictions (implementation‑dependent) to further improve recall at some cost in speed.

In this repository, we load:
- Processor: Owlv2Processor
- Model: Owlv2ForObjectDetection with checkpoint “google/owlv2-large-patch14-ensemble”
- FP16 on CUDA for throughput (especially on A100)

---

## Requirements

- Python 3.9+ recommended
- GPU with CUDA (optional but strongly recommended for speed)
- System packages: 
  - libglib2.0 / libsm6 / libxrender / libxext (often needed for OpenCV image ops; on Colab usually preinstalled)

Python packages:
```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121   # adjust CUDA/cu* tag as needed
pip install transformers timm accelerate
pip install opencv-python pillow pandas numpy tqdm
```

Notes:
- If torchvision.ops.nms import fails on your platform, ensure torchvision and torch versions match your CUDA runtime, or use CPU-only wheels.
- For CPU-only environments, remove or guard FP16 casting and expect slower inference.

---

## How to run

1. Put your images under a single root directory, e.g.:
   - /content/drive/MyDrive/d2_dataset_images

2. Adjust paths and settings in main():
   - IMAGES_DIRECTORY
   - BILLBOARD_OUTPUT_DIR
   - SIGNBOARD_OUTPUT_DIR
   - OUTPUT_CSV
   - MAX_WORKERS (thread count)
   - BATCH_SIZE (used for future batching; current pipeline processes one image per thread)

3. Run:
```bash
python enhanced_billboard_signboard.py
```

4. Outputs:
   - Detected billboard images: BILLBOARD_OUTPUT_DIR
   - Detected signboard images: SIGNBOARD_OUTPUT_DIR
   - CSV summary: OUTPUT_CSV

---

## CSV schema (key columns)

- filename, image_path
- billboard_detected, billboard_confidence
- billboard_open_vocab_detected, billboard_open_vocab_confidence
- billboard_cv_detected, billboard_cv_confidence
- signboard_detected, signboard_confidence
- signboard_open_vocab_detected, signboard_open_vocab_confidence
- signboard_cv_detected, signboard_cv_confidence
- processing_success, error
- billboard_saved_path, signboard_saved_path

---

## Tuning and customization

- Update text queries:
  - self.billboard_queries / self.signboard_queries to focus or broaden detection targets
- Geometry filters:
  - min_area_ratio / min_width_ratio / min_height_ratio / aspect_ratio_range / max_area_ratio
- Thresholds:
  - self.confidence_thresholds['billboard_open_vocab'] (default 0.25)
  - self.confidence_thresholds['signboard_open_vocab'] (default 0.30)
  - self.confidence_thresholds['cv_features'] (default 0.55)
- NMS IoU:
  - Change IoU for stricter or looser merging (default 0.4)

---

## What could be improved

1. True GPU batching for OWLv2
   - Current code calls OWLv2 per image (under threads). On GPU, a proper batch inference (processor on a list of images, single forward pass) will yield better throughput and GPU utilization.
   - Replace ThreadPoolExecutor with a DataLoader-like batching pipeline and torch.cuda.amp.autocast for mixed precision.

2. Alternative SOTA open‑vocabulary detectors
   - Grounding DINO: Strong text-conditioned detector with robust phrase grounding. Pros: excellent performance; Cons: heavier setup, potential CUDA/TensorRT friction, tuning effort.
   - YOLO‑World: Open‑vocabulary extension of YOLO family. Pros: fast; Cons: setup/training nuances and environment alignment needed for best results.
   - OWL‑ViT (older gen): Simpler but generally lower performance than OWLv2.
   - Why not used here: No labeled dataset, desire for minimal setup/time, and need for a reproducible single‑file pipeline. Integrating these models well (with optimal performance) typically needs more engineering and/or data.

3. Text detection + OCR fusion
   - Many signboards/billboards contain text. Integrating PaddleOCR or EasyOCR to re‑score regions could boost precision dramatically.
   - Use OCR confidence and text region density as additional features in the ensemble.

4. Better geometry/shape modeling
   - Use quadrilateral detection (e.g., Hough lines + rectangle fitting) and perspective normalization to reduce false positives in complex scenes.

5. Smarter NMS and calibration
   - Per‑query NMS or Soft‑NMS to keep plausible overlapping boxes.
   - Confidence calibration per class/query (billboard vs. signboard) using a small validation set if available.

6. Visual outputs
   - Optional box overlays and crops (save annotated images and per‑detection JSON) for debugging and qualitative analysis.

7. Metrics and evaluation
   - If/when a small labeled set becomes available, add mAP/precision/recall reporting to measure progress.

---

## Limitations

- Open‑vocabulary ambiguity: Text prompts can match unwanted structures with similar visual/text cues.
- Scene variability: Extreme angles, occlusions, and lighting can reduce accuracy.
- No training/fine‑tuning: Without labeled boxes, we rely on generic knowledge and heuristics.

---

## Acknowledgements

- OWLv2: Google Research (used via Hugging Face Transformers)
- Transformers, timm, accelerate: Hugging Face ecosystem
- torchvision and OpenCV: classic vision foundations
- pandas, tqdm: logging and progress utilities

If you build on this, please consider citing the respective projects and licenses.
