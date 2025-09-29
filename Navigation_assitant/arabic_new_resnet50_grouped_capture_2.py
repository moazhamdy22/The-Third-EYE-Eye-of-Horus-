# --- START OF FILE arabic_new_resnet50_grouped_capture_with_audio.py ---

# --- Imports ---
import cv2
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import models, transforms
import numpy as np
import time
import os
import sys
from arabic_reshaper import reshape
from bidi.algorithm import get_display
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
import queue
from collections import defaultdict, Counter # Use Counter for frequency counting
import logging
import statistics # For potential median calculation
from typing import Optional, Dict, Tuple, List

# Add current directory to Python path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# --- Import Audio Handler ---
try:
    from audio_feedback_vision_assitant import AudioFeedbackHandler
    AUDIO_AVAILABLE = True
except ImportError:
    logging.error("Failed to import AudioFeedbackHandler. Audio feedback disabled.")
    AudioFeedbackHandler = None
    AUDIO_AVAILABLE = False

# --- Configuration ---
CATEGORY_FILE = 'arabic_categories_places365_grouped.txt'
MODEL_PATH = 'resnet50_places365.pth'
CAMERA_INDEX = 1

TOP_N_FOR_GROUPING = 5     # Number of top predictions to consider for grouping
CAPTURE_DURATION_SECONDS = 4.0 # How long to capture frames for
NUM_FRAMES_TO_PROCESS = 10    # Max number of sharp frames to process from the batch
BLUR_THRESHOLD = 50.0        # Laplacian variance threshold for sharpness
MIN_FRAMES_FOR_PROCESSING = 3 # Minimum number of sharp frames needed to proceed

# Confidence thresholds for deciding the final output
GROUP_CONFIDENCE_THRESHOLD = 0.3
SPECIFIC_CONFIDENCE_THRESHOLD = 0.3 # Threshold for specific items if no group dominates

# --- Audio Configuration ---
AUDIO_ENABLED = True  # Master switch for audio
AUDIO_SPEAK_ENGLISH = True
AUDIO_SPEAK_ARABIC = True
AUDIO_COOLDOWN_SECONDS = 5.0 # Minimum time between announcements
AUDIO_INITIAL_DELAY_SECONDS = 2.0 # Delay before first announcement after start

# --- Logging Setup ---
# Use basicConfig if running standalone, or getLogger if part of a larger app
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Use __name__ for the logger

# --- LabelHandler class (remains the same) ---
class LabelHandler(FileSystemEventHandler):
    def __init__(self, update_queue):
        self.update_queue = update_queue

    def on_modified(self, event):
        # Ensure the path comparison is robust
        try:
            if not event.is_directory and os.path.samefile(event.src_path, CATEGORY_FILE):
                logger.info(f"Detected change in {CATEGORY_FILE}, queuing update...")
                self.update_queue.put("update")
        except FileNotFoundError:
             # Handle cases where the file might be deleted/renamed during the check
            logger.warning(f"File {event.src_path} not found during modification check.")
        except Exception as e:
            logger.error(f"Error checking modified file: {e}")


class SceneClassifierBatch:

    def __init__(self, shared_audio_handler=None):
        self.model = None
        self.categories_en = []
        self.categories_ar = []
        self.groups_en = []
        self.groups_ar = []
        self.update_queue = queue.Queue()
        self.file_watcher_thread = None
        self.observer = None
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # --- Audio Related Initialization (SIMPLIFIED like object detection) ---
        self.audio_handler = shared_audio_handler
        self.last_announced_scene: Dict[str, Optional[str]] = {"group": None, "specific": None}
        self.last_announcement_time: float = 0.0
        self.start_time: float = time.monotonic()

        if self.audio_handler:
            logger.info("SceneClassifierBatch is using shared AudioFeedbackHandler instance.")
        else:
            if AUDIO_ENABLED:
                logger.warning("Audio feedback is disabled for SceneClassifierBatch (no handler provided).")

        # --- Load Model & Categories ---
        self.load_model()
        self.load_categories()
        self.setup_file_watcher()

    # --- setup_file_watcher, load_model, load_categories (mostly unchanged) ---
    # Add minor error handling improvements if needed
    def setup_file_watcher(self):
        if self.observer is not None:
             try:
                 self.observer.stop()
                 self.observer.join(timeout=1.0) # Add timeout
             except Exception as e:
                 logger.error(f"Error stopping previous observer: {e}")
        self.observer = Observer()
        handler = LabelHandler(self.update_queue)
        # Ensure watch_dir exists before scheduling
        try:
            watch_dir = os.path.dirname(os.path.abspath(CATEGORY_FILE)) or '.'
            if not os.path.isdir(watch_dir):
                logger.error(f"Watch directory '{watch_dir}' not found. Cannot start file watcher.")
                return

            self.observer.schedule(handler, path=watch_dir, recursive=False)
            self.file_watcher_thread = threading.Thread(target=self.observer.start, daemon=True)
            self.file_watcher_thread.start()
            logger.info(f"Started watching {watch_dir} for changes to {os.path.basename(CATEGORY_FILE)}...")
        except Exception as e:
            logger.error(f"Error starting file watcher for {watch_dir}: {e}")

    def load_model(self):
        logger.info("Loading model...")
        try:
            self.model = models.resnet50(num_classes=365)
            # Use safe loading if possible
            try:
                checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
                logger.info("Loaded model weights with weights_only=True.")
            except (TypeError, AttributeError): # Catch AttributeError for older torch versions
                 logger.warning("PyTorch version might not support 'weights_only=True' or weights file incompatible. Loading without it.")
                 checkpoint = torch.load(MODEL_PATH, map_location='cpu')
            # Check if 'state_dict' key exists
            if 'state_dict' not in checkpoint:
                 logger.error(f"ERROR: 'state_dict' key not found in the checkpoint file: {MODEL_PATH}")
                 exit()
            state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
            self.model.load_state_dict(state_dict)
            self.model.eval()
            logger.info("Model loaded successfully.")
        except FileNotFoundError:
            logger.error(f"FATAL: Model file not found at {MODEL_PATH}")
            exit()
        except Exception as e:
            logger.exception(f"FATAL: Error loading model: {e}") # Use exception for stack trace
            exit()

    def load_categories(self):
        logger.info(f"Loading categories from {CATEGORY_FILE}...")
        new_categories_en, new_categories_ar = [], []
        new_groups_en, new_groups_ar = [], []
        category_count = 0
        try:
            with open(CATEGORY_FILE, encoding='utf-8') as class_file:
                for i, line in enumerate(class_file):
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        specific_en = parts[0].split('/')[-1]
                        group_en = parts[1]
                        group_ar = parts[2]
                        # Handle potential multi-word specific Arabic names
                        specific_ar = ' '.join(parts[3:-1]) # Assume last part is index/number
                        if not specific_ar: # Handle cases with maybe only 4 parts if format varies
                            specific_ar = group_ar # Fallback or specific handling needed?

                        new_categories_en.append(specific_en)
                        new_categories_ar.append(specific_ar)
                        new_groups_en.append(group_en)
                        new_groups_ar.append(group_ar)
                        category_count += 1
                    else:
                         logger.warning(f"Skipping malformed line {i+1} in {CATEGORY_FILE}: '{line.strip()}'")

            # --- Critical: Check if lists are populated before assigning ---
            if not new_categories_en:
                logger.error("No categories were loaded. Check category file format and content.")
                # Keep existing lists or clear them? Clearing might be safer.
                self.categories_en, self.categories_ar = [], []
                self.groups_en, self.groups_ar = [], []
                return # Stop further processing in this function

            self.categories_en = new_categories_en
            self.categories_ar = new_categories_ar
            self.groups_en = new_groups_en
            self.groups_ar = new_groups_ar

            logger.info(f"Successfully loaded {len(self.categories_en)} categories and groups.")
            if category_count != 365: # Informational warning
                 logger.warning(f"Expected 365 categories based on Places365, but loaded {category_count}.")

        except FileNotFoundError:
            logger.error(f"ERROR: Category file not found at {CATEGORY_FILE}")
            self.categories_en, self.categories_ar, self.groups_en, self.groups_ar = [], [], [], []
        except Exception as e:
            logger.exception(f"Error loading categories: {e}") # Log stack trace
            self.categories_en, self.categories_ar, self.groups_en, self.groups_ar = [], [], [], []


    # --- get_font, process_arabic_text (unchanged) ---
    # (Using the corrected versions from the previous step)
    def get_font(self, font_size=24):
        possible_font_paths = [
            "C:/Windows/Fonts/arial.ttf", "/System/Library/Fonts/Arial Unicode.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/tahoma.ttf",
            "C:/Windows/Fonts/times.ttf", "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
            "/Library/Fonts/Arial.ttf", "arial.ttf", "DejaVuSans.ttf",
        ]
        for font_path in possible_font_paths:
            if os.path.exists(font_path):
                try:
                    font = ImageFont.truetype(font_path, font_size)
                    # logger.debug(f"Using font: {font_path}") # Debug log if needed
                    return font
                except Exception as e:
                    logger.warning(f"Found font {font_path} but failed to load: {e}")
                    continue
        logger.warning("Could not find a suitable Unicode font. Using default PIL font.")
        try:
            return ImageFont.load_default(size=font_size)
        except AttributeError:
             logger.warning("Older Pillow? Trying load_default() without size.")
             return ImageFont.load_default()
        except Exception as e:
             logger.error(f"Failed to load even the default PIL font: {e}")
             return None

    def process_arabic_text(self, text):
        if not text: return "" # Handle empty strings
        try:
            # Ensure text is string
            if not isinstance(text, str): text = str(text)
            reshaped_text = reshape(text)
            return get_display(reshaped_text)
        except Exception as e:
            logger.error(f"Error processing Arabic text '{text}': {e}")
            return text # Return original text on error


    # --- calculate_blur_score, select_sharp_frames (unchanged) ---
    def calculate_blur_score(self, frame):
        if frame is None: return 0
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            variance = laplacian.var()
            return variance
        except cv2.error as e:
            logger.warning(f"OpenCV error calculating blur score: {e}")
            return 0

    def select_sharp_frames(self, frames, num_required=NUM_FRAMES_TO_PROCESS, blur_threshold=BLUR_THRESHOLD):
        if not frames:
            logger.warning("No frames provided to select_sharp_frames.")
            return []
        frame_scores = [{'index': i, 'score': self.calculate_blur_score(frame), 'frame': frame}
                        for i, frame in enumerate(frames)]
        sharp_frames = [fs for fs in frame_scores if fs['score'] >= blur_threshold]
        sharp_frames.sort(key=lambda x: x['score'], reverse=True)
        selected = [fs['frame'] for fs in sharp_frames[:num_required]]
        top_score_info = f"Top score: {sharp_frames[0]['score']:.2f}" if sharp_frames else "No sharp frames found"
        logger.info(f"Selected {len(selected)} frames (threshold {blur_threshold}). {top_score_info}")
        return selected


    # --- draw_text_with_pil (using corrected version) ---
    def draw_text_with_pil(self, frame, en_texts, ar_texts):
        # (Using the corrected version from the previous answer with fixed Arabic X position)
        if frame is None or not (en_texts or ar_texts): return frame
        try:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image, "RGBA")
            font_size = 24
            font = self.get_font(font_size)
            if font is None:
                logger.error("Cannot draw text: Font is None.")
                return frame

            try:
                bbox = font.getbbox("Aj") # Pillow >= 10.0
                text_height_estimate = bbox[3] - bbox[1] + 10
            except AttributeError:
                 text_height_estimate = font_size + 8 # Fallback

            num_lines = max(len(en_texts), len(ar_texts))
            box_height = 20 + num_lines * text_height_estimate
            box_width = 600
            box_y_start = 10

            draw.rectangle([(10, box_y_start), (10 + box_width, box_y_start + box_height)], fill=(255, 255, 255, 180))

            y_pos = box_y_start + 10
            en_x = 15
            ar_x = 350 # Fixed X for Arabic

            for i in range(num_lines):
                current_y = y_pos + i * text_height_estimate

                if i < len(en_texts):
                    try:
                       draw.text((en_x, current_y), en_texts[i], font=font, fill=(0, 0, 0, 255))
                    except Exception as e:
                       logger.error(f"Error drawing English text '{en_texts[i]}': {e}")

                if i < len(ar_texts):
                    try:
                        # Ensure ar_texts[i] is valid before processing
                        raw_ar_text = ar_texts[i] if isinstance(ar_texts[i], str) else ""
                        if raw_ar_text:
                             processed_ar_text = self.process_arabic_text(raw_ar_text)
                             draw.text((ar_x, current_y), processed_ar_text, font=font, fill=(0, 0, 0, 255))
                        else:
                             logger.warning(f"Empty or invalid Arabic text received for line {i}")
                    except Exception as e:
                       processed_text_debug = self.process_arabic_text(raw_ar_text) if raw_ar_text else "N/A"
                       logger.error(f"Error drawing Arabic text '{raw_ar_text}' (processed: '{processed_text_debug}'): {e}")


            return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGBA2BGR)

        except Exception as e:
            logger.exception(f"Error in draw_text_with_pil: {e}") # Log stack trace
            return frame


    # --- process_frame_batch (unchanged) ---
    def process_frame_batch(self, frames_to_process):
        if not frames_to_process: return []
        batch_results = []
        # logger.info(f"Processing {len(frames_to_process)} selected frames...") # Can be verbose
        for i, frame in enumerate(frames_to_process):
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                input_img = self.transform(pil_image).unsqueeze(0)

                with torch.no_grad():
                    logit = self.model(input_img)
                probabilities = torch.nn.functional.softmax(logit, dim=1).cpu().numpy()[0]

                group_scores = defaultdict(float)
                indices_in_group = defaultdict(list)
                top_indices = np.argsort(probabilities)[-TOP_N_FOR_GROUPING:][::-1]

                for idx in top_indices:
                    # Check index bounds rigorously
                    if 0 <= idx < len(self.categories_en) and 0 <= idx < len(self.groups_en):
                        group_en = self.groups_en[idx]
                        group_scores[group_en] += probabilities[idx]
                        indices_in_group[group_en].append(idx)
                    else:
                        logger.warning(f"Index {idx} out of bounds for categories/groups list (len={len(self.categories_en)}).")


                sorted_groups = sorted(group_scores.items(), key=lambda item: item[1], reverse=True)

                top_group_name_en = None
                top_group_prob = 0.0
                best_specific_in_group_idx = -1
                if sorted_groups and sorted_groups[0][1] >= GROUP_CONFIDENCE_THRESHOLD:
                    top_group_name_en = sorted_groups[0][0]
                    top_group_prob = sorted_groups[0][1]
                    if top_group_name_en in indices_in_group:
                         max_prob_spec = -1
                         for idx in indices_in_group[top_group_name_en]:
                             if 0 <= idx < len(probabilities) and probabilities[idx] > max_prob_spec:
                                 max_prob_spec = probabilities[idx]
                                 best_specific_in_group_idx = idx

                top_specific_idx = np.argmax(probabilities)
                top_specific_name_en = None
                top_specific_prob = 0.0
                if 0 <= top_specific_idx < len(self.categories_en):
                    top_specific_name_en = self.categories_en[top_specific_idx]
                    top_specific_prob = probabilities[top_specific_idx]
                else:
                     logger.warning(f"Top specific index {top_specific_idx} out of bounds.")


                result = {
                    'group_en': top_group_name_en, 'group_prob': top_group_prob,
                    'best_specific_in_group_idx': best_specific_in_group_idx,
                    'top_specific_en': top_specific_name_en, 'top_specific_prob': top_specific_prob,
                }
                batch_results.append(result)

            except Exception as e:
                logger.error(f"Error processing frame {i} in the batch: {e}")
                continue
        return batch_results

    # --- aggregate_and_decide (unchanged, but returns processed AR text) ---
    def aggregate_and_decide(self, batch_results) -> Tuple[str, str, str, str]:
        """Analyzes batch results. Returns EN Group, Processed AR Group, EN Specific, Processed AR Specific."""
        if not batch_results:
            return "Unknown", self.process_arabic_text("غير معروف"), "Unknown", self.process_arabic_text("غير معروف")

        group_counter = Counter()
        specific_counter = Counter()
        confident_group_predictions = []

        for result in batch_results:
            if result.get('group_en') and result.get('group_prob', 0) >= GROUP_CONFIDENCE_THRESHOLD:
                group_counter[result['group_en']] += 1
                confident_group_predictions.append(result) # Store full result for specific lookup

            # Count confident specifics independently
            if result.get('top_specific_en') and result.get('top_specific_prob', 0) >= SPECIFIC_CONFIDENCE_THRESHOLD:
                 specific_counter[result['top_specific_en']] += 1
                 # No need to store confident_specific_predictions if only counting


        final_group_en: Optional[str] = None
        final_specific_en: Optional[str] = None

        # --- Decision Logic ---
        if group_counter:
            # Check if the most common group is significantly frequent
            most_common_group, group_freq = group_counter.most_common(1)[0]
            # Example: Require the group to appear in > 1/3 of confident frames
            if group_freq > len(confident_group_predictions) / 3:
                final_group_en = most_common_group

                # Find most common specific *within* the chosen dominant group's frames
                specifics_in_dominant_group = Counter()
                for res in confident_group_predictions:
                     if res.get('group_en') == final_group_en:
                         idx = res.get('best_specific_in_group_idx', -1)
                         if 0 <= idx < len(self.categories_en):
                              specifics_in_dominant_group[self.categories_en[idx]] += 1

                if specifics_in_dominant_group:
                    # Check if the top specific in this group is frequent enough
                    top_specific_in_group, spec_freq_in_group = specifics_in_dominant_group.most_common(1)[0]
                    # Example: Require specific to be in > half the frames the group was dominant
                    if spec_freq_in_group > group_freq / 2:
                         final_specific_en = top_specific_in_group


        # Priority 2: If no dominant group, find the most frequent *confident* specific overall
        if not final_group_en and specific_counter:
             most_common_specific, spec_freq = specific_counter.most_common(1)[0]
             # Example: Require overall specific to appear in > 1/3 of all batch results
             if spec_freq > len(batch_results) / 3:
                 final_specific_en = most_common_specific


        # --- Get AR names (use raw names for lookup, then process for return) ---
        raw_group_ar = "غير معروف"
        raw_specific_ar = "غير معروف"

        if final_group_en:
            try:
                idx = self.groups_en.index(final_group_en)
                raw_group_ar = self.groups_ar[idx]
            except (ValueError, IndexError): pass

        if final_specific_en:
             try:
                 idx = self.categories_en.index(final_specific_en)
                 raw_specific_ar = self.categories_ar[idx]
             except (ValueError, IndexError): pass

        # Process AR text just before returning
        final_group_ar_processed = self.process_arabic_text(raw_group_ar)
        final_specific_ar_processed = self.process_arabic_text(raw_specific_ar)

        # logger.info(f"Aggregation: Confident Groups: {group_counter}") # Can be verbose
        # logger.info(f"Aggregation: Confident Specifics: {specific_counter}") # Can be verbose
        logger.info(f"Decision: Final Group='{final_group_en}', Final Specific='{final_specific_en}'")

        # Ensure None is converted to "Unknown"
        group_en_out = final_group_en or "Unknown"
        specific_en_out = final_specific_en or "Unknown"

        return group_en_out, final_group_ar_processed, specific_en_out, final_specific_ar_processed

    def _generate_scene_audio_message(self, group_en, group_ar_processed, specific_en, specific_ar_processed) -> str:
        """Creates a more descriptive and human-like combined audio message."""
        message_en = ""
        message_ar = ""

        # --- Clean names for audio (remove underscores) ---
        group_en_clean = group_en.replace('_', ' ')
        specific_en_clean = specific_en.replace('_', ' ')
        # Apply replacement to the already processed Arabic text
        group_ar_clean = group_ar_processed.replace('_', ' ')
        specific_ar_clean = specific_ar_processed.replace('_', ' ')

        # --- English Message Construction ---
        if group_en != "Unknown" and specific_en != "Unknown":
            # Both known: Describe general then specific
            message_en = f"Okay, it seems you are in an {group_en_clean} environment. More specifically, this looks like a {specific_en_clean}."
            # Alternative: f"Alright, the surroundings feel like an {group_en_clean} area. This part strongly resembles a {specific_en_clean}."

        elif group_en != "Unknown":
            # Only group known: Describe the general setting
            message_en = f"Okay, the environment appears to be {group_en_clean}."
            # Alternative: f"The general scene seems to be {group_en_clean}."

        elif specific_en != "Unknown":
            # Only specific known (less common, but possible): Focus on the specific identification
            message_en = f"Okay, analyzing the details... this looks like a {specific_en_clean}."
            # Alternative: f"This area resembles a {specific_en_clean}."
        else:
            # Neither is known
            message_en = "I'm having trouble identifying the current scene."
            # Alternative: "The scene around you is currently unclear."

        # --- Arabic Message Construction (Mirroring the English Structure) ---
        # Pre-process fixed phrases (or process inline if preferred)
        okay_ar = self.process_arabic_text("حسنًا")
        it_seems_you_are_in_ar = self.process_arabic_text("يبدو أنك في بيئة") # Seems you are in an environment [type]
        environment_ar = self.process_arabic_text("بيئة")
        more_specifically_ar = self.process_arabic_text("وبشكل أكثر تحديدًا")
        this_looks_like_ar = self.process_arabic_text("هذا يبدو مثل") # This looks like [specific]
        appears_to_be_ar = self.process_arabic_text("يبدو أنه") # Appears to be [type]
        analyzing_details_ar = self.process_arabic_text("جارٍ تحليل التفاصيل")
        resembles_ar = self.process_arabic_text("يشبه") # Resembles [specific]
        trouble_identifying_ar = self.process_arabic_text("أواجه صعوبة في تحديد المشهد الحالي")
        scene_unclear_ar = self.process_arabic_text("المشهد من حولك غير واضح حاليًا")


        if group_en != "Unknown" and specific_en != "Unknown":
            # Use cleaned, processed Arabic names
            message_ar = f"{okay_ar}، {it_seems_you_are_in_ar} {group_ar_clean}. {more_specifically_ar}، {this_looks_like_ar} {specific_ar_clean}."
            # Alt: f"{okay_ar}، {self.process_arabic_text('المحيط يبدو مثل منطقة')} {group_ar_clean}. {self.process_arabic_text('هذا الجزء يشبه بقوة')} {specific_ar_clean}."

        elif group_en != "Unknown":
            message_ar = f"{okay_ar}، {self.process_arabic_text('البيئة تبدو')} {group_ar_clean}."
            # Alt: f"{self.process_arabic_text('المشهد العام يبدو')} {group_ar_clean}."

        elif specific_en != "Unknown":
            message_ar = f"{okay_ar}، {analyzing_details_ar}... {this_looks_like_ar} {specific_ar_clean}."
            # Alt: f"{self.process_arabic_text('هذه المنطقة تشبه')} {specific_ar_clean}."
        else:
            message_ar = trouble_identifying_ar
            # Alt: scene_unclear_ar

        # --- Combine based on flags ---
        final_message_parts = []
        if AUDIO_SPEAK_ENGLISH and message_en:
            final_message_parts.append(message_en)
        if AUDIO_SPEAK_ARABIC and message_ar:
            separator = " " # Use a simple space separator for better flow between languages maybe? Or keep ". "?
            if final_message_parts: # Add separator only if English part exists
                 final_message_parts.append(separator + message_ar)
            else:
                 final_message_parts.append(message_ar)


        final_message = "".join(final_message_parts).strip()

        # Return a default message if somehow both parts ended up empty
        return final_message if final_message else "Scene information unavailable."


    def _handle_audio_feedback(self, final_group_en, final_group_ar, final_specific_en, final_specific_ar):
        """Checks state, cooldown, and triggers audio if necessary."""
        if not self.audio_handler:
            return

        now = time.monotonic()
        current_scene_state = {"group": final_group_en, "specific": final_specific_en}

        # --- Check Initial Delay ---
        is_initial_period = (now - self.start_time) < AUDIO_INITIAL_DELAY_SECONDS
        if is_initial_period and self.last_announcement_time == 0:
            return # Don't announce immediately on start

        # --- Check for State Change ---
        changed = (current_scene_state["group"] != self.last_announced_scene["group"] or
                   current_scene_state["specific"] != self.last_announced_scene["specific"])

        # --- Check Cooldown ---
        cooldown_active = (now - self.last_announcement_time < AUDIO_COOLDOWN_SECONDS)

        # --- Decision to Speak ---
        should_speak = False
        if changed and not cooldown_active:
             should_speak = True
             logger.info("Scene change detected, triggering audio.")
        # Handle first announcement after initial delay correctly
        elif self.last_announcement_time == 0 and not is_initial_period:
             should_speak = True
             logger.info("First announcement after initial delay.")

        if should_speak:
            message = self._generate_scene_audio_message(
                final_group_en, final_group_ar, final_specific_en, final_specific_ar
            )
            if message: # Only speak if a message was generated
                logger.info(f"AUDIO: Speaking: '{message}'")
                self.audio_handler.speak(message)  # Remove priority parameter
                self.last_announced_scene = current_scene_state.copy() # Update last announced state
                self.last_announcement_time = now
            else:
                 logger.warning("Generated audio message was empty, not speaking.")

    def run(self):
        if not self.categories_en:
            logger.error("FATAL: No categories loaded. Check category file. Cannot run.")
            if self.audio_handler:
                self.audio_handler.speak("Error: Scene categories not loaded. Cannot start.")
            return

        cap = cv2.VideoCapture(CAMERA_INDEX,cv2.CAP_DSHOW) # Use cv2.CAP_DSHOW for Windows DirectShow
        if not cap.isOpened():
            logger.error(f"FATAL: Could not open camera {CAMERA_INDEX}")
            if self.audio_handler:
                self.audio_handler.speak("Error: Camera not found. Scene classification cannot start.")
            return

        logger.info("Starting batch classification loop...")
        logger.info("Press '0' to quit")

        if self.audio_handler:
            self.audio_handler.speak("Scene classification is now active. I will describe your surroundings.")

        # ... (last_frame_display initialization remains) ...
        last_frame_display = None
        display_text_en = ["Initializing..."]
        display_text_ar = ["جار التهيئة..."] # Raw Arabic

        ret_init, init_frame = cap.read()
        if ret_init:
            last_frame_display = cv2.resize(init_frame, (1280, 720)) # Resize to fit display
        else:
            last_frame_display = np.zeros((720, 1280, 3), dtype=np.uint8)

        is_running = True
        user_initiated_quit = False

        try: # Outer try for the main loop
            while is_running:
                try: # Inner try for per-iteration processing
                    # ... (Category File Update Check - remains the same) ...
                    try:
                        update_signal = self.update_queue.get_nowait()
                        if update_signal == "update":
                            self.load_categories()
                    except queue.Empty: pass
                    except Exception as e: logger.error(f"Error checking update queue: {e}")


                    # --- Capture Phase ---
                    captured_frames = []
                    start_capture_time = time.monotonic()
                    while time.monotonic() - start_capture_time < CAPTURE_DURATION_SECONDS:
                        ret, frame = cap.read()
                        if ret:
                            frame_resized = cv2.resize(frame, (1280, 720))
                            captured_frames.append(frame_resized.copy())
                            last_frame_display = frame_resized
                        else:
                            logger.warning("Failed to capture frame during capture phase.")
                            time.sleep(0.05)

                        frame_with_text = self.draw_text_with_pil(last_frame_display, display_text_en, display_text_ar)
                        cv2.imshow('Scene Classification (Batch)', frame_with_text)

                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('0'):
                            logger.info("Quit signal received during capture.")
                            is_running = False
                            user_initiated_quit = True
                            break

                    if not is_running: 
                        break

                    # ... (Selection Phase - remains the same) ...
                    if not captured_frames:
                        logger.warning("No frames captured. Skipping processing.")
                        time.sleep(0.5)
                        continue
                    selected_frames = self.select_sharp_frames(captured_frames)
                    if len(selected_frames) < MIN_FRAMES_FOR_PROCESSING:
                        logger.warning(f"Too few sharp frames ({len(selected_frames)} < {MIN_FRAMES_FOR_PROCESSING}). Skipping.")
                        new_en = ["Too few sharp frames"]
                        new_ar = ["إطارات غير واضحة"]
                        if new_en != display_text_en or new_ar != display_text_ar:
                            display_text_en = new_en
                            display_text_ar = new_ar
                            frame_with_text = self.draw_text_with_pil(last_frame_display, display_text_en, display_text_ar)
                            cv2.imshow('Scene Classification (Batch)', frame_with_text)
                            cv2.waitKey(1)
                        time.sleep(0.5)
                        continue

                    # ... (Processing Phase - remains the same) ...
                    batch_results = self.process_frame_batch(selected_frames)

                    # ... (Aggregation and Decision Phase - remains the same) ...
                    final_group_en, final_group_ar_processed_for_speech, \
                    final_specific_en, final_specific_ar_processed_for_speech = self.aggregate_and_decide(batch_results)

                    # ... (Output Phase (Terminal & Display Text Prep) - Arabic display text fixed) ...
                    logger.info(f"Scene: G='{final_group_en}', S='{final_specific_en}' / AR_G_Speech='{final_group_ar_processed_for_speech}', AR_S_Speech='{final_specific_ar_processed_for_speech}'")
                    display_text_en_new = []
                    display_text_ar_new = []
                    raw_group_ar_disp = "غير معروف"
                    raw_specific_ar_disp = "غير معروف"
                    if final_group_en != "Unknown":
                        try: idx = self.groups_en.index(final_group_en); raw_group_ar_disp = self.groups_ar[idx]
                        except (ValueError, IndexError): pass
                    if final_specific_en != "Unknown":
                        try: idx = self.categories_en.index(final_specific_en); raw_specific_ar_disp = self.categories_ar[idx]
                        except (ValueError, IndexError): pass
                    if final_group_en != "Unknown":
                        display_text_en_new.append(f"Group: {final_group_en}")
                        display_text_ar_new.append(f"{raw_group_ar_disp} :مجموعة")
                        if final_specific_en != "Unknown":
                            display_text_en_new.append(f" Likely: {final_specific_en}")
                            display_text_ar_new.append(f"{raw_specific_ar_disp} :مرجح")
                    elif final_specific_en != "Unknown":
                        display_text_en_new.append(f"Scene: {final_specific_en}")
                        display_text_ar_new.append(f"{raw_specific_ar_disp} :مشهد")
                    else:
                        display_text_en_new.append("Scene: Unknown")
                        display_text_ar_new.append("مشهد غير معروف")
                    if display_text_en_new != display_text_en or display_text_ar_new != display_text_ar:
                        display_text_en = display_text_en_new
                        display_text_ar = display_text_ar_new

                    self._handle_audio_feedback(final_group_en, final_group_ar_processed_for_speech,
                                              final_specific_en, final_specific_ar_processed_for_speech)

                    frame_with_text = self.draw_text_with_pil(last_frame_display, display_text_en, display_text_ar)
                    cv2.imshow('Scene Classification (Batch)', frame_with_text)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('0'):
                        logger.info("Quit signal received after processing.")
                        is_running = False
                        user_initiated_quit = True
                        break

                except KeyboardInterrupt:
                    logger.info("KeyboardInterrupt received during main loop. Stopping...")
                    is_running = False
                    user_initiated_quit = True
                    break
                except Exception as e_loop:
                    logger.exception(f"An unexpected error occurred in the main processing loop: {e_loop}")
                    is_running = False
                    break

        finally:
            # --- Cleanup (SIMPLIFIED like object detection) ---
            logger.info("Starting cleanup...")
            
            if self.audio_handler and user_initiated_quit:
                exit_message = "Scene classification has been stopped. Thank you for using this feature."
                logger.info("User initiated quit. Playing exit message.")
                try:
                    self.audio_handler.speak(exit_message)
                    time.sleep(2.0)
                except Exception as e:
                    logger.error(f"Error playing exit message: {e}")

            if cap is not None:
                cap.release()
                logger.info("Video capture released.")

            if self.observer:
                try:
                    self.observer.stop()
                    if self.file_watcher_thread and self.file_watcher_thread.is_alive():
                         self.file_watcher_thread.join(timeout=1.0)
                    logger.info("File watcher stopped.")
                except Exception as e:
                    logger.error(f"Error stopping file watcher: {e}")
            
            try:
                cv2.destroyWindow('Scene Classification (Batch)')
                logger.info("OpenCV window 'Scene Classification (Batch)' destroyed.")
            except Exception as e:
                 logger.warning(f"Error destroying OpenCV window 'Scene Classification (Batch)': {e}")

            logger.info("SceneClassifierBatch run method finished gracefully.")

# --- Main Function (SIMPLIFIED like object detection) ---
def main(shared_audio_handler_external=None):
    """
    Main function for scene classification.
    
    Args:
        shared_audio_handler_external: Optional AudioFeedbackHandler instance
    """
    print("\n=== 🎯 Scene Classification Assistant ===")
    print("Starting up... Please wait.")

    if not shared_audio_handler_external:
        print("❌ AudioFeedbackHandler is required for scene classification.")
        logger.error("AudioFeedbackHandler is required for scene classification.")
        return

    # Pass the shared audio handler directly to the classifier
    try:
        classifier = SceneClassifierBatch(shared_audio_handler=shared_audio_handler_external)
        classifier.run()
    except Exception as e:
        logger.exception("Unhandled exception in main execution of SceneClassifier.")
        if shared_audio_handler_external:
            shared_audio_handler_external.speak("An error occurred in scene classification.")
    finally:
        print("\n=== 👋 Thank you for using Scene Classification Assistant ===")
        logger.info("SceneClassifier main function finished.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create audio handler for standalone mode
    test_audio_handler = None
    try:
        if AUDIO_ENABLED and AUDIO_AVAILABLE:
            test_audio_handler = AudioFeedbackHandler()
            if not test_audio_handler.engine:
                print("❌ Failed to initialize audio. Scene classification cannot start.")
                exit()
        else:
            print("❌ Audio not available. Scene classification requires audio feedback.")
            exit()
            
        main(shared_audio_handler_external=test_audio_handler)
        
    except Exception as e:
        print(f"❌ Standalone test failed: {e}")
    finally:
        if test_audio_handler:
            test_audio_handler.stop()
        cv2.destroyAllWindows()
        print("Standalone test finished.")

# --- END OF FILE arabic_new_resnet50_grouped_capture_with_audio.py ---