import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time
from typing import List, Tuple, Dict
import logging
from tqdm import tqdm
import threading
import json
import shutil
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Optional: if you still want torchvision NMS
from torchvision.ops import nms

# OWLv2 for enhanced open-vocabulary object detection
from transformers import Owlv2Processor, Owlv2ForObjectDetection

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedBillboardSignboardDetector:
    def __init__(self, device=None, batch_size=8):
        """Initialize the enhanced billboard/signboard detection system with OWLv2 Large"""
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size  # Process multiple images at once for A100
        print(f"🚀 Using device: {self.device}")
        print(f"📊 Batch size: {self.batch_size}")

        # Check GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🎯 GPU Memory: {gpu_memory:.1f} GB")

        # Load models
        self.models = {}
        self.load_models()

        # Enhanced detection confidence thresholds for OWLv2 Large
        self.confidence_thresholds = {
            'billboard_open_vocab': 0.25,    # Lower threshold for better recall with OWLv2
            'signboard_open_vocab': 0.30,    # Slightly higher for signboards
            'cv_features': 0.55              # CV features threshold
        }

        # Separate queries for billboards and signboards
        self.billboard_queries = [
            "large advertising billboard",
            "roadside billboard",
            "highway billboard",
            "street billboard",
            "digital billboard",
            "outdoor advertising hoarding",
            "commercial billboard",
            "billboard advertisement"
        ]

        self.signboard_queries = [
            "shop signboard",
            "store sign",
            "business sign",
            "street signboard",
            "directional sign",
            "information signboard",
            "warning sign",
            "road sign"
        ]

        # Enhanced geometry filters
        # Billboard filters (larger structures)
        self.billboard_filters = {
            'min_area_ratio': 0.015,        # >= 1.5% of image
            'min_width_ratio': 0.12,        # width >= 12% of image width
            'min_height_ratio': 0.06,       # height >= 6% of image height
            'aspect_ratio_range': (1.2, 8.0),  # Wide rectangles
            'max_area_ratio': 0.85          # Avoid whole-image detections
        }

        # Signboard filters (smaller, more varied structures)
        self.signboard_filters = {
            'min_area_ratio': 0.005,        # >= 0.5% of image
            'min_width_ratio': 0.05,        # width >= 5% of image width
            'min_height_ratio': 0.03,       # height >= 3% of image height
            'aspect_ratio_range': (0.3, 12.0), # More flexible aspect ratios
            'max_area_ratio': 0.70          # Avoid whole-image detections
        }

    def load_models(self):
        """Load OWLv2 Large for enhanced zero-shot detection"""
        print("📦 Loading OWLv2 Large model (enhanced open-vocabulary detection)...")
        try:
            # Using the large model for better performance on A100
            model_name = "google/owlv2-large-patch14-ensemble"
            self.models['owlv2_processor'] = Owlv2Processor.from_pretrained(model_name)
            self.models['owlv2'] = Owlv2ForObjectDetection.from_pretrained(model_name)
            self.models['owlv2'].to(self.device).eval()

            # Enable mixed precision for faster inference on A100
            if self.device == 'cuda':
                self.models['owlv2'] = self.models['owlv2'].half()  # Use FP16

            print("✅ OWLv2 Large loaded successfully with FP16 optimization")
        except Exception as e:
            print(f"❌ Failed to load OWLv2 Large: {e}")
            print("   Tip: pip install transformers timm accelerate")
            self.models['owlv2_processor'] = None
            self.models['owlv2'] = None

    def detect_with_enhanced_open_vocab(self, image_path: str) -> Dict:
        """Enhanced detection using OWLv2 Large with separate billboard and signboard detection"""
        if self.models.get('owlv2') is None or self.models.get('owlv2_processor') is None:
            return {
                'billboard_detected': False, 'billboard_confidence': 0.0, 'billboard_detections': [],
                'signboard_detected': False, 'signboard_confidence': 0.0, 'signboard_detections': []
            }

        try:
            image = Image.open(image_path).convert("RGB")
            W, H = image.size

            processor = self.models['owlv2_processor']
            model = self.models['owlv2']

            # Combine all queries for single inference
            all_queries = self.billboard_queries + self.signboard_queries

            # Prepare inputs
            inputs = processor(text=[all_queries], images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Convert to half precision if using FP16
            if self.device == 'cuda' and next(model.parameters()).dtype == torch.float16:
                if 'pixel_values' in inputs:
                    inputs['pixel_values'] = inputs['pixel_values'].half()

            with torch.no_grad():
                outputs = model(**inputs)

            target_sizes = torch.tensor([[H, W]], device=self.device)
            results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes)
            result = results[0]

            boxes = result["boxes"]
            scores = result["scores"]
            labels = result["labels"]

            # Separate billboard and signboard detections
            billboard_candidates = []
            signboard_candidates = []

            for i in range(len(scores)):
                score = float(scores[i].item())
                label_idx = int(labels[i].item())

                x1, y1, x2, y2 = [float(v) for v in boxes[i].tolist()]
                bw = max(0.0, x2 - x1)
                bh = max(0.0, y2 - y1)

                if bw <= 0 or bh <= 0:
                    continue

                area_ratio = (bw * bh) / (W * H + 1e-6)
                aspect_ratio = (bw / bh) if bh > 0 else 0

                # Determine if it's a billboard or signboard query
                is_billboard_query = label_idx < len(self.billboard_queries)

                if is_billboard_query:
                    # Billboard detection with billboard-specific filters
                    if (score >= self.confidence_thresholds['billboard_open_vocab'] and
                        self.billboard_filters['min_area_ratio'] <= area_ratio <= self.billboard_filters['max_area_ratio'] and
                        (bw / W) >= self.billboard_filters['min_width_ratio'] and
                        (bh / H) >= self.billboard_filters['min_height_ratio'] and
                        self.billboard_filters['aspect_ratio_range'][0] <= aspect_ratio <= self.billboard_filters['aspect_ratio_range'][1]):

                        class_name = self.billboard_queries[label_idx]
                        billboard_candidates.append({
                            'class': class_name,
                            'confidence': score,
                            'bbox': [x1, y1, x2, y2],
                            'area_ratio': area_ratio,
                            'aspect_ratio': aspect_ratio
                        })
                else:
                    # Signboard detection with signboard-specific filters
                    signboard_label_idx = label_idx - len(self.billboard_queries)
                    if (score >= self.confidence_thresholds['signboard_open_vocab'] and
                        self.signboard_filters['min_area_ratio'] <= area_ratio <= self.signboard_filters['max_area_ratio'] and
                        (bw / W) >= self.signboard_filters['min_width_ratio'] and
                        (bh / H) >= self.signboard_filters['min_height_ratio'] and
                        self.signboard_filters['aspect_ratio_range'][0] <= aspect_ratio <= self.signboard_filters['aspect_ratio_range'][1]):

                        class_name = self.signboard_queries[signboard_label_idx]
                        signboard_candidates.append({
                            'class': class_name,
                            'confidence': score,
                            'bbox': [x1, y1, x2, y2],
                            'area_ratio': area_ratio,
                            'aspect_ratio': aspect_ratio
                        })

            # Apply NMS to both categories separately
            def apply_nms(candidates):
                if not candidates:
                    return []
                b = torch.tensor([c['bbox'] for c in candidates], dtype=torch.float32)
                s = torch.tensor([c['confidence'] for c in candidates], dtype=torch.float32)
                keep = nms(b, s, iou_threshold=0.4).tolist()
                return [candidates[i] for i in keep]

            billboard_candidates = apply_nms(billboard_candidates)
            signboard_candidates = apply_nms(signboard_candidates)

            # Calculate results
            billboard_detected = len(billboard_candidates) > 0
            signboard_detected = len(signboard_candidates) > 0

            billboard_confidence = max([c['confidence'] for c in billboard_candidates], default=0.0)
            signboard_confidence = max([c['confidence'] for c in signboard_candidates], default=0.0)

            return {
                'billboard_detected': billboard_detected,
                'billboard_confidence': billboard_confidence,
                'billboard_detections': billboard_candidates,
                'signboard_detected': signboard_detected,
                'signboard_confidence': signboard_confidence,
                'signboard_detections': signboard_candidates
            }

        except Exception as e:
            logger.warning(f"Enhanced open-vocab detection failed for {image_path}: {e}")
            return {
                'billboard_detected': False, 'billboard_confidence': 0.0, 'billboard_detections': [],
                'signboard_detected': False, 'signboard_confidence': 0.0, 'signboard_detections': []
            }

    def detect_with_cv_features(self, image_path: str) -> Dict:
        """Enhanced CV-based feature detection for both billboards and signboards"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return {
                    'billboard_detected': False, 'billboard_confidence': 0.0,
                    'signboard_detected': False, 'signboard_confidence': 0.0,
                    'features': []
                }

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape

            # Enhanced edge detection
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 40, 120, apertureSize=3)

            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            billboard_features = []
            signboard_features = []

            for contour in contours:
                area = cv2.contourArea(contour)
                if area <= 100:  # Filter very small contours
                    continue

                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                if len(approx) >= 4:  # Rectangular-ish shapes
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    area_ratio = (w * h) / (width * height + 1e-6)

                    # Classify as billboard or signboard based on size and aspect ratio
                    if (self.billboard_filters['aspect_ratio_range'][0] <= aspect_ratio <= self.billboard_filters['aspect_ratio_range'][1] and
                        area_ratio >= self.billboard_filters['min_area_ratio'] and
                        w >= width * self.billboard_filters['min_width_ratio'] and
                        h >= height * self.billboard_filters['min_height_ratio']):

                        billboard_features.append({
                            'type': 'rectangular_shape',
                            'area_ratio': float(area_ratio),
                            'aspect_ratio': float(aspect_ratio),
                            'bbox': [int(x), int(y), int(w), int(h)]
                        })

                    elif (self.signboard_filters['aspect_ratio_range'][0] <= aspect_ratio <= self.signboard_filters['aspect_ratio_range'][1] and
                          area_ratio >= self.signboard_filters['min_area_ratio'] and
                          w >= width * self.signboard_filters['min_width_ratio'] and
                          h >= height * self.signboard_filters['min_height_ratio']):

                        signboard_features.append({
                            'type': 'rectangular_shape',
                            'area_ratio': float(area_ratio),
                            'aspect_ratio': float(aspect_ratio),
                            'bbox': [int(x), int(y), int(w), int(h)]
                        })

            # Text region detection (both types often contain text)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            connected = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            text_contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            text_regions = sum(1 for c in text_contours if cv2.contourArea(c) > 500)

            # Calculate confidence scores
            billboard_confidence = min(len(billboard_features) * 0.4 + (text_regions > 0) * 0.3, 1.0)
            signboard_confidence = min(len(signboard_features) * 0.5 + (text_regions > 0) * 0.2, 1.0)

            billboard_detected = billboard_confidence > self.confidence_thresholds['cv_features']
            signboard_detected = signboard_confidence > self.confidence_thresholds['cv_features']

            return {
                'billboard_detected': billboard_detected,
                'billboard_confidence': float(billboard_confidence),
                'signboard_detected': signboard_detected,
                'signboard_confidence': float(signboard_confidence),
                'features': billboard_features + signboard_features,
                'billboard_features': len(billboard_features),
                'signboard_features': len(signboard_features),
                'text_regions': text_regions
            }

        except Exception as e:
            logger.warning(f"CV feature detection failed for {image_path}: {e}")
            return {
                'billboard_detected': False, 'billboard_confidence': 0.0,
                'signboard_detected': False, 'signboard_confidence': 0.0,
                'features': []
            }

    def ensemble_prediction(self, open_vocab_result: Dict, cv_result: Dict) -> Dict:
        """Enhanced ensemble prediction for both billboards and signboards"""
        # Weights favoring the powerful OWLv2 model
        weights = {'open_vocab': 0.8, 'cv_features': 0.2}

        # Billboard ensemble
        billboard_confidence = (
            open_vocab_result['billboard_confidence'] * weights['open_vocab'] +
            cv_result['billboard_confidence'] * weights['cv_features']
        )

        billboard_detected = (
            open_vocab_result['billboard_detected'] or
            (cv_result['billboard_detected'] and billboard_confidence > 0.6)
        )

        # Signboard ensemble
        signboard_confidence = (
            open_vocab_result['signboard_confidence'] * weights['open_vocab'] +
            cv_result['signboard_confidence'] * weights['cv_features']
        )

        signboard_detected = (
            open_vocab_result['signboard_detected'] or
            (cv_result['signboard_detected'] and signboard_confidence > 0.6)
        )

        return {
            'billboard_detected': billboard_detected,
            'billboard_confidence': float(billboard_confidence),
            'signboard_detected': signboard_detected,
            'signboard_confidence': float(signboard_confidence),
            'individual_results': {
                'open_vocab': open_vocab_result,
                'cv_features': cv_result
            }
        }

    def process_single_image(self, image_path: str) -> Dict:
        """Process a single image for both billboard and signboard detection"""
        try:
            open_vocab_result = self.detect_with_enhanced_open_vocab(image_path)
            cv_result = self.detect_with_cv_features(image_path)
            final_result = self.ensemble_prediction(open_vocab_result, cv_result)

            return {
                'image_path': image_path,
                'success': True,
                'result': final_result
            }

        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return {
                'image_path': image_path,
                'success': False,
                'error': str(e),
                'result': {
                    'billboard_detected': False, 'billboard_confidence': 0.0,
                    'signboard_detected': False, 'signboard_confidence': 0.0
                }
            }

def save_detected_image(image_path, output_dir, detection_type, confidence):
    """Save detected image to appropriate output directory"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(image_path)
        base_name, ext = os.path.splitext(filename)
        new_filename = f"{base_name}_{detection_type}_conf_{confidence:.2f}{ext}"
        output_path = os.path.join(output_dir, new_filename)
        shutil.copy2(image_path, output_path)
        return True, output_path
    except Exception as e:
        logger.error(f"Error saving {detection_type} image {image_path}: {e}")
        return False, None

def process_images_batch(detector: EnhancedBillboardSignboardDetector, image_paths: List[str],
                        billboard_output_dir: str, signboard_output_dir: str, max_workers: int = 6) -> List[Dict]:
    """Enhanced batch processing with separate output folders"""
    results = []
    successful_detections = 0
    failed_detections = 0
    billboards_saved = 0
    signboards_saved = 0

    lock = threading.Lock()

    def update_progress(result):
        nonlocal successful_detections, failed_detections, billboards_saved, signboards_saved

        with lock:
            if result['success']:
                successful_detections += 1

                # Save billboard if detected
                if result['result']['billboard_detected']:
                    success, saved_path = save_detected_image(
                        result['image_path'],
                        billboard_output_dir,
                        'billboard',
                        result['result']['billboard_confidence']
                    )
                    if success:
                        billboards_saved += 1
                        result['billboard_saved_path'] = saved_path

                # Save signboard if detected
                if result['result']['signboard_detected']:
                    success, saved_path = save_detected_image(
                        result['image_path'],
                        signboard_output_dir,
                        'signboard',
                        result['result']['signboard_confidence']
                    )
                    if success:
                        signboards_saved += 1
                        result['signboard_saved_path'] = saved_path
            else:
                failed_detections += 1

    start_time = time.time()
    progress_bar = tqdm(total=len(image_paths), desc="🔍 Detecting billboards & signboards", unit="img")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {
            executor.submit(detector.process_single_image, path): path
            for path in image_paths
        }

        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                result = future.result()
                update_progress(result)
                results.append(result)

                # Progress indicators
                billboard_status = "🎪" if result.get('result', {}).get('billboard_detected', False) else ""
                signboard_status = "🪧" if result.get('result', {}).get('signboard_detected', False) else ""
                status = billboard_status + signboard_status if billboard_status or signboard_status else "❌"

                billboard_conf = result.get('result', {}).get('billboard_confidence', 0.0)
                signboard_conf = result.get('result', {}).get('signboard_confidence', 0.0)

                progress_bar.set_postfix({
                    'Success': successful_detections,
                    'Failed': failed_detections,
                    'Billboards': billboards_saved,
                    'Signboards': signboards_saved,
                    'Last': f"{status} (B:{billboard_conf:.2f}, S:{signboard_conf:.2f})"
                })

                progress_bar.update(1)

            except Exception as e:
                logger.error(f"Exception processing {path}: {e}")
                results.append({
                    'image_path': path,
                    'success': False,
                    'error': str(e),
                    'result': {'billboard_detected': False, 'billboard_confidence': 0.0,
                              'signboard_detected': False, 'signboard_confidence': 0.0}
                })
                update_progress({'success': False})
                progress_bar.update(1)

    progress_bar.close()

    end_time = time.time()
    duration = end_time - start_time

    billboard_count = sum(1 for r in results if r.get('result', {}).get('billboard_detected', False))
    signboard_count = sum(1 for r in results if r.get('result', {}).get('signboard_detected', False))
    total_detections = billboard_count + signboard_count

    print(f"\n{'='*80}")
    print(f"🎯 ENHANCED DETECTION SUMMARY (OWLv2 Large)")
    print(f"{'='*80}")
    print(f"📁 Total images processed: {len(image_paths)}")
    print(f"✅ Successful detections: {successful_detections}")
    print(f"❌ Failed detections: {failed_detections}")
    print(f"🎪 Images with billboards: {billboard_count}")
    print(f"🪧 Images with signboards: {signboard_count}")
    print(f"📊 Total detections: {total_detections}")
    print(f"💾 Billboards saved to drive: {billboards_saved}")
    print(f"💾 Signboards saved to drive: {signboards_saved}")
    print(f"📈 Billboard detection rate: {billboard_count/len(image_paths)*100:.1f}%")
    print(f"📈 Signboard detection rate: {signboard_count/len(image_paths)*100:.1f}%")
    print(f"📈 Overall detection rate: {total_detections/len(image_paths)*100:.1f}%")
    print(f"⏱️  Total processing time: {duration:.1f} seconds")
    print(f"⚡ Average time per image: {duration/len(image_paths):.2f} seconds")
    print(f"🚀 Processing speed: {len(image_paths)/duration:.1f} images/second")
    print(f"{'='*80}")

    return results

def save_results_to_csv(results: List[Dict], output_path: str):
    """Save enhanced detection results to CSV file"""
    csv_data = []

    for result in results:
        image_path = result['image_path']
        filename = os.path.basename(image_path)

        if result['success']:
            detection_result = result['result']
            individual_results = detection_result.get('individual_results', {})
            open_vocab = individual_results.get('open_vocab', {})
            cv_features = individual_results.get('cv_features', {})

            csv_row = {
                'filename': filename,
                'image_path': image_path,

                # Billboard results
                'billboard_detected': detection_result.get('billboard_detected', False),
                'billboard_confidence': detection_result.get('billboard_confidence', 0.0),
                'billboard_open_vocab_detected': open_vocab.get('billboard_detected', False),
                'billboard_open_vocab_confidence': open_vocab.get('billboard_confidence', 0.0),
                'billboard_cv_detected': cv_features.get('billboard_detected', False),
                'billboard_cv_confidence': cv_features.get('billboard_confidence', 0.0),

                # Signboard results
                'signboard_detected': detection_result.get('signboard_detected', False),
                'signboard_confidence': detection_result.get('signboard_confidence', 0.0),
                'signboard_open_vocab_detected': open_vocab.get('signboard_detected', False),
                'signboard_open_vocab_confidence': open_vocab.get('signboard_confidence', 0.0),
                'signboard_cv_detected': cv_features.get('signboard_detected', False),
                'signboard_cv_confidence': cv_features.get('signboard_confidence', 0.0),

                # Processing info
                'processing_success': True,
                'error': '',
                'billboard_saved_path': result.get('billboard_saved_path', ''),
                'signboard_saved_path': result.get('signboard_saved_path', '')
            }
        else:
            csv_row = {
                'filename': filename,
                'image_path': image_path,
                'billboard_detected': False, 'billboard_confidence': 0.0,
                'billboard_open_vocab_detected': False, 'billboard_open_vocab_confidence': 0.0,
                'billboard_cv_detected': False, 'billboard_cv_confidence': 0.0,
                'signboard_detected': False, 'signboard_confidence': 0.0,
                'signboard_open_vocab_detected': False, 'signboard_open_vocab_confidence': 0.0,
                'signboard_cv_detected': False, 'signboard_cv_confidence': 0.0,
                'processing_success': False,
                'error': result.get('error', 'Unknown error'),
                'billboard_saved_path': '', 'signboard_saved_path': ''
            }

        csv_data.append(csv_row)

    df = pd.DataFrame(csv_data)
    df.to_csv(output_path, index=False)

    print(f"💾 Results saved to: {output_path}")

    # Enhanced summary
    total_images = len(df)
    successful_processing = int(df['processing_success'].sum())
    detected_billboards = int(df['billboard_detected'].sum())
    detected_signboards = int(df['signboard_detected'].sum())
    saved_billboards = int(df['billboard_saved_path'].str.len().gt(0).sum())
    saved_signboards = int(df['signboard_saved_path'].str.len().gt(0).sum())

    print(f"\n📊 DETAILED RESULTS SUMMARY:")
    print(f"📁 Total images: {total_images}")
    print(f"✅ Successfully processed: {successful_processing}")
    print(f"🎪 Billboards detected: {detected_billboards}")
    print(f"🪧 Signboards detected: {detected_signboards}")
    print(f"💾 Billboards saved: {saved_billboards}")
    print(f"💾 Signboards saved: {saved_signboards}")
    if successful_processing > 0:
        print(f"📈 Billboard detection rate: {detected_billboards/successful_processing*100:.1f}%")
        print(f"📈 Signboard detection rate: {detected_signboards/successful_processing*100:.1f}%")

def main():
    """Main function to run enhanced billboard/signboard detection"""
    # Enhanced Configuration for A100 GPU
    IMAGES_DIRECTORY = "/content/drive/MyDrive/d2_dataset_images"  # Updated input directory
    BILLBOARD_OUTPUT_DIR = "/content/drive/MyDrive/billboard_detected"  # Separate folder for billboards
    SIGNBOARD_OUTPUT_DIR = "/content/drive/MyDrive/signboard_detected"  # Separate folder for signboards
    OUTPUT_CSV = "/content/drive/MyDrive/enhanced_detection_results.csv"
    MAX_WORKERS = 8  # Increased for A100
    BATCH_SIZE = 8   # Batch processing for efficiency
    MAX_IMAGES = None  # Process all images

    print("🎪🪧 Starting Enhanced Billboard & Signboard Detection System...")
    print(f"🤖 Using OWLv2 Large (Patch14) - Optimized for A100 GPU")
    print(f"📂 Images Directory: {IMAGES_DIRECTORY}")
    print(f"🎪 Billboard Output Directory: {BILLBOARD_OUTPUT_DIR}")
    print(f"🪧 Signboard Output Directory: {SIGNBOARD_OUTPUT_DIR}")
    print(f"📊 Output CSV: {OUTPUT_CSV}")
    print(f"⚡ Max Workers: {MAX_WORKERS}")
    print(f"📦 Batch Size: {BATCH_SIZE}")
    print("=" * 80)

    # Create output directories
    os.makedirs(BILLBOARD_OUTPUT_DIR, exist_ok=True)
    os.makedirs(SIGNBOARD_OUTPUT_DIR, exist_ok=True)

    # Initialize enhanced detector with A100 optimizations
    detector = EnhancedBillboardSignboardDetector(batch_size=BATCH_SIZE)

    # Find all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.JPG', '.JPEG', '.PNG'}
    image_paths = []

    if not os.path.exists(IMAGES_DIRECTORY):
        print(f"❌ Images directory does not exist: {IMAGES_DIRECTORY}")
        return

    print(f"🔍 Scanning for images in {IMAGES_DIRECTORY}...")
    for root, dirs, files in os.walk(IMAGES_DIRECTORY):
        for file in files:
            if any(file.endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(root, file))

    print(f"🖼️  Found {len(image_paths)} images")

    if MAX_IMAGES and MAX_IMAGES < len(image_paths):
        image_paths = image_paths[:MAX_IMAGES]
        print(f"📏 Limited to first {MAX_IMAGES} images for processing")

    if not image_paths:
        print("❌ No images found in the specified directory!")
        return

    # Process images with enhanced detection
    results = process_images_batch(
        detector,
        image_paths,
        BILLBOARD_OUTPUT_DIR,
        SIGNBOARD_OUTPUT_DIR,
        max_workers=MAX_WORKERS
    )

    # Save enhanced results
    save_results_to_csv(results, OUTPUT_CSV)

    print(f"\n✨ Enhanced detection completed successfully!")
    print(f"🎪 Billboard images saved to: {BILLBOARD_OUTPUT_DIR}")
    print(f"🪧 Signboard images saved to: {SIGNBOARD_OUTPUT_DIR}")
    print(f"📊 Detailed results saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
