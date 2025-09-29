# File: ocr_combined.py
"""Combined OCR Script with Direct Model Access"""

import cv2
import os
import pytesseract
import threading
from datetime import datetime
import warnings
import pyttsx3
from gtts import gTTS
import tempfile
from pygame import mixer
from langdetect import detect, DetectorFactory
import numpy as np
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
# Add these imports at the top with other imports
# from TTS.api import TTS
import time

# --- EDITED: Fix the translation system import ---
# Use relative import since translation_system.py is in the same OCR directory
from translation_system import FrameTranslator

# --- EDITED: Remove the duplicate import that's already handled above ---
from audio_feedback_vision_assitant import AudioFeedbackHandler


# Fix random language detection issue
DetectorFactory.seed = 0

# Suppress warnings
warnings.filterwarnings("ignore")

# Initialize logging
def log_info(message):
    """Log information with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[OCR {timestamp}] {message}")

class OCRProcessor:
    def __init__(self, cap=None, shared_audio_handler=None):
        self.cap = cap  # Accept camera object from outside
        
        # --- EDITED: Store the shared audio handler ---
        self.audio_handler = shared_audio_handler
        
        # Remove camera initialization from here
        self.initialize_models()
        self.initialize_tts()
        self.epsilon_factors = [0.02, 0.03, 0.04, 0.05]
        self.min_contour_area = 5000
        self.reference_ratio = 1.414
        self.ratio_tolerance = 0.2
        
        # Initialize camera as None - will be set by main system
        self.running = True
        
        # Initialize FrameTranslator once
        api_key = "AIzaSyCay_6tu-NW-ZAAJtSp_iBV3PwEAkndIH4"
        self.translator = FrameTranslator(api_key)
        log_info("translation initialized successfully")

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect    

    def initialize_tts(self):
        """Initialize text-to-speech engines"""
        # Initialize pyttsx3 for feedback comments
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        
        # Try to initialize Coqui TTS for English, but handle potential errors
        self.coqui_available = False
        try:
            log_info("Loading Coqui TTS model...")
            # Import TTS only if needed to avoid errors if not installed properly
            try:
                import torch
                from TTS.api import TTS
                self.coqui_tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
                log_info("Coqui TTS loaded successfully")
                self.coqui_available = True
            except ImportError as import_err:
                log_info(f"TTS library not installed: {import_err}")
                self.coqui_available = False
            except Exception as model_err:
                log_info(f"Error loading Coqui TTS model: {model_err}")
                self.coqui_available = False
        except Exception as e:
            log_info(f"Error loading Coqui TTS, will use gTTS for all languages: {e}")
            self.coqui_available = False
    def initialize_models(self): 
        """Initialize the OCR models."""
        log_info("Configuring Tesseract-OCR...")
        pytesseract.pytesseract.tesseract_cmd = r"D:\4th Biomedical year finallll\graduation project  Smart Glass for Blind People\Code\OCR\Model\tesseract-ocr-w64-setup-5.5.0.20241111\tesseract.exe"
        os.environ["TESSDATA_PREFIX"] = r"D:\4th Biomedical year finallll\graduation project  Smart Glass for Blind People\Code\OCR\Model\tesseract-ocr-w64-setup-5.5.0.20241111\tessdata"
        log_info("Tesseract-OCR configured")
    
        # Load the Qari model from local path
        log_info("Loading Qari OCR model from local directory...")
        try:
            local_model_path = r"local_model"
            local_processor_path = r"local_processor"
            
            # Check if local model exists
            if os.path.exists(local_model_path) and os.path.exists(local_processor_path):
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(local_model_path, torch_dtype="auto").to("cuda")
                self.processor = AutoProcessor.from_pretrained(local_processor_path)
                log_info("Qari OCR model loaded successfully from local directory")
            else:
                log_info("Local model not found. Downloading from Hugging Face...")
                model_name = "NAMAA-Space/Qari-OCR-0.1-VL-2B-Instruct"
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto").to("cuda")
                self.processor = AutoProcessor.from_pretrained(model_name)
                log_info("Qari OCR model loaded successfully from Hugging Face")
                
                # Save model locally for future use
                log_info("Saving model locally for offline use...")
                self.model.save_pretrained(local_model_path)
                self.processor.save_pretrained(local_processor_path)
                log_info("Model saved locally")
                
            # Set model type attribute
            self.model_type = "Qari"
        except Exception as e:
            log_info(f"Error loading Qari model: {e}")
            log_info("Will use Tesseract OCR as fallback")
            self.model = None
            self.processor = None
            self.model_type = None

    def speak_text(self, text, language="en"):
        """Use the shared audio handler to speak feedback messages."""
        if self.audio_handler:
            # Use the robust, thread-safe handler from the main system
            self.audio_handler.speak(text)
        else:
            # This is a fallback for testing the file by itself (standalone mode)
            print(f"AUDIO_FALLBACK (pyttsx3): {text}")
            try:
                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"Standalone TTS Error: {e}")

    def speak_extracted_text(self, text, language):
        """
        Speak extracted text using Coqui TTS for English and gTTS for Arabic.
        """
        if not text.strip():
            self.speak_text("No text was extracted", "en")
            return
            
        try:
            print(f"Speaking extracted text in {language}")
            self.speak_text(f"Reading extracted {language} text", "en")
            
            if language == "Arabic":
                # Use gTTS for Arabic
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False, 
                    suffix='.mp3',
                    dir=tempfile.gettempdir()
                )
                temp_file.close()
                tts = gTTS(text=text, lang='ar')
                tts.save(temp_file.name)
                self._play_audio_file(temp_file.name)
            else:
                # For English, use Coqui TTS
                if self.coqui_available:
                    try:
                        temp_file = tempfile.NamedTemporaryFile(
                            delete=False, 
                            suffix='.wav',
                            dir=tempfile.gettempdir()
                        )
                        temp_file.close()
                        
                        # Generate audio directly with text
                        self.coqui_tts.tts_to_file(
                            text=text,
                            file_path=temp_file.name
                        )
                        
                        if os.path.exists(temp_file.name) and os.path.getsize(temp_file.name) > 0:
                            self._play_audio_file(temp_file.name)
                        else:
                            raise Exception("Audio file generation failed")
                            
                    except Exception as e:
                        print(f"Coqui TTS failed, falling back to gTTS: {e}")
                        # Fallback to gTTS
                        temp_file = tempfile.NamedTemporaryFile(
                            delete=False, 
                            suffix='.mp3',
                            dir=tempfile.gettempdir()
                        )
                        temp_file.close()
                        tts = gTTS(text=text, lang='en')
                        tts.save(temp_file.name)
                        self._play_audio_file(temp_file.name)
                else:
                    # Use gTTS if Coqui not available
                    temp_file = tempfile.NamedTemporaryFile(
                        delete=False, 
                        suffix='.mp3',
                        dir=tempfile.gettempdir()
                    )
                    temp_file.close()
                    tts = gTTS(text=text, lang='en')
                    tts.save(temp_file.name)
                    self._play_audio_file(temp_file.name)
                    
        except Exception as e:
            print(f"Error in text-to-speech: {e}")
            self.speak_text(f"Error reading text", "en")
        finally:
            # Cleanup handled in the _play_audio_file method
            pass

    def _play_audio_file(self, file_path):
        """
        Helper method to play audio files with proper error handling and cleanup.
        """
        temp_file_path = file_path  # Store file path for cleanup
        try:
            # Initialize pygame mixer
            mixer.init()
            
            # Load and play the audio
            mixer.music.load(temp_file_path)
            mixer.music.play()
            
            # Wait for audio to finish
            while mixer.music.get_busy():
                time.sleep(0.1)
                
            # Properly unload and quit
            mixer.music.unload()
            mixer.quit()
        except Exception as play_error:
            print(f"Error playing audio: {play_error}")
            
            # If pygame fails, try an alternative approach with sounddevice and scipy
            try:
                import sounddevice as sd
                from scipy.io import wavfile
                
                # Only works for WAV files
                if temp_file_path.endswith('.wav'):
                    sample_rate, data = wavfile.read(temp_file_path)
                    sd.play(data, sample_rate)
                    sd.wait()
                else:
                    # For MP3, fall back to pyttsx3
                    raise Exception("Non-WAV format with sounddevice fallback")
            except:
                # Final fallback to pyttsx3
                self.speak_text(f"Error reading text: {str(play_error)}", "en")
        finally:
            # Cleanup temp file with retries
            try:
                # Give a moment for any file operations to complete
                time.sleep(0.2)
                for _ in range(3):
                    try:
                        if os.path.exists(temp_file_path):
                            os.unlink(temp_file_path)
                        break
                    except PermissionError:
                        time.sleep(0.1)
            except Exception as cleanup_error:
                print(f"Cleanup failed: {cleanup_error}")

    
    def preprocess_image(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Increase blur kernel size and adjust parameters
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adjust adaptive threshold parameters
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 21, 4)
        
        # Use larger kernel for morphological operations
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        
        # Adjust Canny parameters and add median blur
        median = cv2.medianBlur(blurred, 5)
        edges = cv2.Canny(median, 50, 150)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        return {'gray': gray, 'blurred': blurred, 'thresh': thresh, 
                'dilated': dilated, 'edges': edges}
    

    def find_document_contours(self, processed_images):
            # Add morphological operations to clean up edges
            kernel = np.ones((5, 5), np.uint8)
            morph_edges = cv2.morphologyEx(processed_images['edges'], cv2.MORPH_CLOSE, kernel)
            
            contours_thresh, _ = cv2.findContours(
                processed_images['dilated'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contours_canny, _ = cv2.findContours(
                morph_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Filter contours by area more strictly
            filtered_contours = []
            for contour in (contours_thresh + contours_canny):
                area = cv2.contourArea(contour)
                if area > self.min_contour_area:
                    # Add perimeter check
                    peri = cv2.arcLength(contour, True)
                    if area/peri > 50:  # Filter based on area/perimeter ratio
                        filtered_contours.append(contour)
            
            return sorted(filtered_contours, key=cv2.contourArea, reverse=True) 

    def detect_document_edges(self, frame):
        """Enhanced document edge detection with improved crop window size control"""
        processed_images = self.preprocess_image(frame)
        height, width = frame.shape[:2]
        
        # Refined area constraints - adjust these values to control crop size
        min_area = (width * height) * 0.12  # Increase minimum to 15% of frame
        max_area = (width * height) * 0.85  # Decrease maximum to 80% of frame
        
        # Refined document size expectations (adjust based on your typical documents)
        expected_min_area_ratio = 0.15  # Document should take at least 20% of frame
        expected_max_area_ratio = 0.80  # Document should take at most 60% of frame
        
        # Find contours with enhanced filtering
        all_contours = self.find_document_contours(processed_images)
        
        document_found = False
        document_contour = None
        best_score = 0
        used_epsilon = None
        is_accurate = False
        crop_complete = False
        crop_size_appropriate = False  # New flag to track appropriate crop size
        
        for epsilon_factor in self.epsilon_factors:
            for contour in all_contours[:10]:
                area = cv2.contourArea(contour)
                area_ratio = area / (width * height)
                
                # Apply stricter area filtering
                if area < min_area or area > max_area:
                    continue
                
                # Check if contour size is within expected document size range
                if expected_min_area_ratio <= area_ratio <= expected_max_area_ratio:
                    crop_size_appropriate = True
                else:
                    # Allow but penalize contours outside ideal range
                    crop_size_appropriate = False
                
                peri = cv2.arcLength(contour, True)
                epsilon = epsilon_factor * peri
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) == 4:
                    rect = self.order_points(approx.reshape(4, 2))
                    (tl, tr, br, bl) = rect
                    
                    # Calculate dimensions
                    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
                    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
                    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
                    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
                    
                    # Check for minimum width and height
                    min_doc_width = width * 0.2  # Document should be at least 20% of frame width
                    min_doc_height = height * 0.2  # Document should be at least 20% of frame height
                    
                    if width_a < min_doc_width or width_b < min_doc_width or height_a < min_doc_height or height_b < min_doc_height:
                        continue
                    
                    max_width = max(int(width_a), int(width_b))
                    max_height = max(int(height_a), int(height_b))
                    
                    # Check aspect ratio against reference
                    current_ratio = max_height / float(max_width)
                    ratio_diff = abs(current_ratio - self.reference_ratio)
                    is_accurate = ratio_diff <= self.ratio_tolerance
                    
                    # Check aspect ratio (typical document ratios)
                    aspect_ratio = max_width / float(max_height)
                    if not (0.5 <= aspect_ratio <= 2.0):
                        continue
                    
                    # Check if the contour is too close to frame edges
                    padding = 10  # pixels of padding
                    points_near_edge = 0
                    for point in rect:
                        x, y = point
                        if (x < padding or x > width - padding or 
                            y < padding or y > height - padding):
                            points_near_edge += 1
                    
                    crop_complete = points_near_edge <= 1
                    
                    # Calculate distance to center of frame
                    center_x, center_y = width / 2, height / 2
                    contour_center_x = (tl[0] + br[0]) / 2
                    contour_center_y = (tl[1] + br[1]) / 2
                    
                    # Calculate normalized distance to center (0 = at center, 1 = at edge)
                    center_dist = np.sqrt((contour_center_x - center_x)**2 + (contour_center_y - center_y)**2)
                    max_dist = np.sqrt((width/2)**2 + (height/2)**2)
                    center_score = 1 - min(center_dist / max_dist, 1.0)
                    
                    # Calculate content density in the contour area
                    mask = np.zeros(processed_images['edges'].shape, dtype=np.uint8)
                    cv2.drawContours(mask, [approx], -1, 255, -1)
                    edge_pixels = cv2.countNonZero(cv2.bitwise_and(processed_images['edges'], mask))
                    density_score = min(edge_pixels / (area + 1) * 500, 1.0)  # Normalize density score
                    
                    # Calculate size penalty for too large areas
                    size_penalty = 1.0
                    if area_ratio > expected_max_area_ratio:
                        # Penalize contours that are too large
                        size_penalty = expected_max_area_ratio / area_ratio
                    
                    # Comprehensive scoring system
                    current_score = (
                        center_score * 0.3 +            # Reward centered documents
                        density_score * 0.3 +           # Reward areas with document-like content
                        (1.0 if is_accurate else 0.7) * 0.2 +  # Reward accurate aspect ratios
                        (1.0 if crop_complete else 0.7) * 0.2  # Reward complete crops
                    ) * size_penalty                   # Penalize oversized crops
                    
                    if current_score > best_score:
                        best_score = current_score
                        document_contour = approx
                        document_found = True
                        used_epsilon = epsilon_factor
        
        return {
            'success': document_found,
            'contour': document_contour,
            'epsilon': used_epsilon,
            'preprocessing': processed_images,
            'score': best_score if document_found else 0,
            'is_accurate': is_accurate,
            'crop_complete': crop_complete,
            'crop_size_appropriate': crop_size_appropriate
        }
    

    def check_boundary_contrast(self, frame, rect):
        """Check contrast around document boundary to ensure full document is captured"""
        # Convert to grayscale if not already
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        # Sample points inside and outside of each edge
        contrast_scores = []
        padding = 5  # pixels to check inside/outside
        
        # Check each edge of the rectangle
        edges = [
            (rect[0], rect[1]),  # top edge
            (rect[1], rect[2]),  # right edge
            (rect[2], rect[3]),  # bottom edge
            (rect[3], rect[0])   # left edge
        ]
        
        for p1, p2 in edges:
            # Get several sample points along this edge
            samples = 5
            for i in range(1, samples+1):
                # Point on the edge
                t = i / (samples+1)
                edge_x = int((1-t) * p1[0] + t * p2[0])
                edge_y = int((1-t) * p1[1] + t * p2[1])
                
                # Calculate perpendicular direction (normalized)
                edge_vec = [p2[0] - p1[0], p2[1] - p1[1]]
                length = np.sqrt(edge_vec[0]**2 + edge_vec[1]**2)
                if length == 0:
                    continue
                    
                perp_vec = [-edge_vec[1]/length, edge_vec[0]/length]
                
                # Sample inside and outside points
                in_x = int(edge_x - perp_vec[0] * padding)
                in_y = int(edge_y - perp_vec[1] * padding)
                out_x = int(edge_x + perp_vec[0] * padding)
                out_y = int(edge_y + perp_vec[1] * padding)
                
                # Ensure points are within the image
                h, w = gray.shape[:2]
                if (0 <= in_x < w and 0 <= in_y < h and 
                    0 <= out_x < w and 0 <= out_y < h):
                    # Get intensity values
                    in_val = gray[in_y, in_x]
                    out_val = gray[out_y, out_x]
                    
                    # Calculate contrast
                    contrast = abs(int(in_val) - int(out_val)) / 255.0
                    contrast_scores.append(contrast)
        
        # Return average contrast score or 0.5 if we couldn't calculate
        return sum(contrast_scores) / len(contrast_scores) if contrast_scores else 0.5    
    

    def detect_document(self, frame):
        """Main document detection function with improved crop size control"""
        display = frame.copy()
        
        try:
            # Step 1: Detect document edges
            detection_result = self.detect_document_edges(frame)
            
            if detection_result['success']:
                # Draw the contour outline (optional)
                cv2.drawContours(display, [detection_result['contour']], -1, (0, 255, 0), 2)

                # Create a mask for the detected document
                mask = np.zeros_like(display, dtype=np.uint8)
                cv2.fillPoly(mask, [detection_result['contour']], color=(0, 255, 0))

                # Blend the mask with the display image for transparency
                alpha = 0.3  # Transparency factor (0.0 - 1.0)
                cv2.addWeighted(mask, alpha, display, 1 - alpha, 0, display)

                # Determine message based on detection results
                if detection_result['is_accurate'] and detection_result['crop_complete'] and detection_result['crop_size_appropriate']:
                    status_msg = "Document found! (Crop complete)"
                    status_color = (0, 255, 0)  # Green
                elif not detection_result['crop_size_appropriate']:
                    status_msg = "Document found! (Crop area too large - move closer)"
                    status_color = (0, 0, 255)  # Red
                elif not detection_result['crop_complete']:
                    status_msg = "Document found! (Adjust position - document may be cut off)"
                    status_color = (0, 165, 255)  # Orange
                else:
                    status_msg = "Document found! (Not accurate - please adjust)"
                    status_color = (0, 165, 255)  # Orange
                    
                cv2.putText(display, status_msg, 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                        status_color, 2)
            else:
                cv2.putText(display, "No document detected - Try adjusting lighting/position", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            return {
                'success': detection_result['success'] and 
                        detection_result['is_accurate'] and 
                        detection_result['crop_complete'] and 
                        detection_result['crop_size_appropriate'],
                'display': display,
                'contour': detection_result['contour'],
                'preprocessing': np.hstack((
                    detection_result['preprocessing']['thresh'],
                    detection_result['preprocessing']['dilated'],
                    detection_result['preprocessing']['edges']
                ))
            }
                
        except Exception as e:
            cv2.putText(display, f"Error: {str(e)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return {'success': False, 'display': display, 'error': str(e)}    

    def crop_document(self, frame, contour):
        """Crops and warps the detected document with improved size constraints"""
        if contour is None:
            return None
        
        # Get original image dimensions
        frame_height, frame_width = frame.shape[:2]
        
        rect = self.order_points(contour.reshape(4, 2))
        (tl, tr, br, bl) = rect
        
        # Calculate dimensions
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        # Calculate contour area and frame area
        contour_area = maxWidth * maxHeight
        frame_area = frame_width * frame_height
        
        # If contour is too large compared to frame (over 70%), adjust it
        if (contour_area > frame_area * 0.7):
            # Calculate the scaling factor needed
            scale_factor = np.sqrt(0.6 * frame_area / contour_area)
            
            # Adjust points to shrink the contour
            center_x = sum(p[0] for p in rect) / 4
            center_y = sum(p[1] for p in rect) / 4
            
            adjusted_rect = []
            for point in rect:
                x, y = point
                # Move the point toward the center
                new_x = center_x + (x - center_x) * scale_factor
                new_y = center_y + (y - center_y) * scale_factor
                adjusted_rect.append([new_x, new_y])
            
            rect = np.array(adjusted_rect, dtype="float32")
            (tl, tr, br, bl) = rect
            
            # Recalculate dimensions
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))
            
            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))
        
        # Add small padding
        maxWidth += 10
        maxHeight += 10
        
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")
        
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))
        
        return warped     

    def enhance_document_image(self, image):
        """Enhances the document image using advanced processing techniques"""
        if len(image.shape) == 3:
            # Convert to LAB color space for better color enhancement
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            
            # Merge channels
            enhanced_lab = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            # Sharpen the image
            kernel = np.array([[-1,-1,-1],
                             [-1, 9,-1],
                             [-1,-1,-1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            
            # Adjust contrast and brightness
            alpha = 1.2  # Contrast control
            beta = 10    # Brightness control
            enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)
            
            return enhanced
        else:
            # For grayscale images
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
            
            # Apply bilateral filter to reduce noise while preserving edges
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            # Apply adaptive thresholding
            enhanced = cv2.adaptiveThreshold(
                enhanced, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 
                11, 2
            )
            
            return enhanced 

    def detect_language_from_image(self, image):
        """Detect if text in image is Arabic or English."""
        log_info("Detecting language from image")
        
        # First get the document detection result
        detection_result = self.detect_document(image)

        
        # If we successfully detected a document, use the cropped and enhanced version
        if (detection_result['success'] and 
            detection_result.get('contour') is not None):
            cropped_document = self.crop_document(image, detection_result['contour'])
            if cropped_document is not None:
                # Use the enhanced version for OCR
                enhanced_image = self.enhance_document_image(cropped_document)
                # Now use pytesseract on the actual image
                text = pytesseract.image_to_string(enhanced_image, lang="ara+eng", config="--psm 6")
            else:
                # If cropping failed, use original image
                text = pytesseract.image_to_string(image, lang="ara+eng", config="--psm 6")
        else:
            # If no document detected, use original image
            text = pytesseract.image_to_string(image, lang="ara+eng", config="--psm 6")
        
        if not text.strip():
            log_info("No text detected for language identification")
            return "Unknown"
        
        try:
            language = detect(text)
            if language == "ar":
                log_info("Arabic language detected")
                return "Arabic"
            elif language == "en":
                log_info("English language detected")
                return "English"
            return "Unknown"
        except Exception as e:
            log_info(f"Language detection error: {e}")
            return "Unknown"
    
    def detect_language(self, image):
        """Method to detect language - calls detect_language_from_image"""
        return self.detect_language_from_image(image)
    
    def extract_text(self, image, language):
        """Extract text from image based on detected language."""
        log_info(f"Extracting {language} text")
        
        # Always try to use Qari model first
        try:
            return self.process_image_with_qari(image, language)
        except Exception as e:
            log_info(f"Qari model failed, using Tesseract as fallback: {e}")
            # Don't use detect_document here, just pass the image directly to pytesseract
            # since we've already processed/enhanced it
            lang_code = "ara" if language == "Arabic" else "eng"
            return pytesseract.image_to_string(image, lang=lang_code, config="--psm 6")
    
    def process_image_with_qari(self, image, language):
        """Process the image with the loaded model for text extraction."""
        if self.model is None or self.processor is None:
            log_info("No deep learning model available, falling back to Tesseract")
            processed_image = self.detect_document(image)
            lang_code = "ara" if language == "Arabic" else "eng"
            return pytesseract.image_to_string(processed_image, lang=lang_code, config="--psm 6")
        
        log_info(f"Extracting {language} text with Qari model")
        
        # Create a temporary directory with explicit permissions
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, "temp_image.jpg")
        
        try:
            # Save the image to the temporary file
            cv2.imwrite(temp_file_path, image)
            
            # Check if file was created successfully
            if not os.path.exists(temp_file_path):
                raise FileNotFoundError(f"Failed to create temporary file at {temp_file_path}")
            
            # Process with Qari model
            prompt = f"Extract all {language} text from this image."
            
            messages = [{"role": "user", "content": [
                {"type": "image", "image": f"file://{temp_file_path}"}, 
                {"type": "text", "text": prompt}
            ]}]

            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            
            # Import here to avoid issues if the module is not available
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")
            
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=1000, 
                do_sample=True,
                temperature=0.2,
                top_p=0.9,
                top_k=50
            )

            output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            remove_phrases = ["system", "You are a helpful assistant.", "user", prompt, "assistant"]
            for phrase in remove_phrases:
                output_text = output_text.replace(phrase, "").strip()
            return output_text
        
        except Exception as e:
            log_info(f"Error in model processing: {e}")
            raise e
        
        finally:
            # Clean up temporary files
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)

    def process_document(self, enhanced_image, processing_complete_flag=None):
        """
        Process the document: detect language, extract text, speak.
        Optionally pass a mutable flag (e.g., a dict) to signal completion.
        """
        try:
            print("Starting language detection...")
            detected_language = self.detect_language(enhanced_image)
            print(f"Detected language: {detected_language}")
            self.speak_text(f"Detected {detected_language} text", "en")
            
            if detected_language in ["Arabic", "English"]:
                print("Starting text extraction...")
                extracted_text = self.extract_text(enhanced_image, detected_language)
                extracted_text = extracted_text.replace('\n', ' ').strip()
                
                if extracted_text.strip():
                    print(f"\nExtracted {detected_language} text:\n{extracted_text}")
                    self.speak_extracted_text(extracted_text, detected_language)
                else:
                    print("No text could be extracted")
                    self.speak_text("No text could be extracted", "en")
            
            self.speak_text("Processing complete. Move document to scan again.", "en")
        except Exception as e:
            print(f"Error during processing: {e}")
            self.speak_text("Error processing document", "en")
        finally:
            if processing_complete_flag is not None:
                processing_complete_flag['value'] = False

    def prepare_and_process_document(self, frame, processing_complete_flag, processing_thread_holder):
        """
        Prepare the document (detect, crop, enhance) and start processing in a thread.
        processing_complete_flag: dict with key 'value' for thread-safe flag.
        processing_thread_holder: dict with key 'thread' to hold the thread object.
        """
        process_frame = frame.copy()
        detection_result = self.detect_document(process_frame)
        if detection_result['success'] and detection_result.get('contour') is not None:
            cropped = self.crop_document(process_frame, detection_result['contour'])
        else:
            cropped = process_frame
        enhanced = self.enhance_document_image(cropped)
        scale_factor = 2
        new_width = int(enhanced.shape[1] * scale_factor)
        new_height = int(enhanced.shape[0] * scale_factor)
        enlarged_enhanced = cv2.resize(enhanced, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        # cv2.imshow("enhanced Document", enlarged_enhanced)
        # cv2.waitKey(1000)
        # cv2.destroyWindow("Preprocessing")
        # cv2.destroyWindow("enhanced Document")
        processing_complete_flag['value'] = True
        processing_thread = threading.Thread(
            target=self.process_document,
            args=(enhanced.copy(), processing_complete_flag)
        )
        processing_thread.start()
        processing_thread_holder['thread'] = processing_thread

    def capture_and_process(self):
        """Open camera, detect language, then extract text."""
        # --- EDITED: Add welcome and instruction messages ---
        self.speak_text("OCR system activated. Hold a document steady for automatic scanning, press 0 to exit, press 1 to translate, or press Enter for manual capture.")
        
    
        self.running = True
        last_contour = None
        stable_start_time = None
        stable_duration = 1.0  # Time in seconds to hold document steady
        processing_complete_flag = {'value': False}
        processing_thread_holder = {'thread': None}
        switch_mode = None  # Track if user wants to switch mode
        window_focused = False  # Track if window has been focused

        try:
            if not hasattr(self, 'cap') or self.cap is None:
                self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                
            if not self.cap.isOpened():
                print("Could not open camera")
                self.speak_text("Camera not available", "en")
                return False

            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame")
                    continue

                detection_result = self.detect_document(frame)
                current_time = time.time()

                # Display the frame
                cv2.imshow("OCR Mode", detection_result['display'])
                
                # Force window to foreground on first frame
                if not window_focused:
                    cv2.setWindowProperty("OCR Mode", cv2.WND_PROP_TOPMOST, 1)
                    cv2.setWindowProperty("OCR Mode", cv2.WND_PROP_TOPMOST, 0)  # Remove topmost after focusing
                    window_focused = True
                
                current_contour = detection_result.get('contour')
                
                # Document detection and processing logic (auto mode)
                if detection_result['success'] and current_contour is not None and not processing_complete_flag['value']:
                    if last_contour is None:
                        last_contour = current_contour
                        stable_start_time = current_time
                        self.speak_text("Document detected, hold steady", "en")
                    
                    if current_time - stable_start_time >= stable_duration:
                        try:
                            self.prepare_and_process_document(frame, processing_complete_flag, processing_thread_holder)
                        except Exception as e:
                            print(f"Error during processing: {e}")
                            self.speak_text("Error processing document", "en")

                elif not detection_result['success']:
                    last_contour = None
                    stable_start_time = None
                    processing_complete_flag['value'] = False
                
                key = cv2.waitKey(1) & 0xFF

                # Manual capture with Enter key
                if key == 13:  # Enter key
                    try:
                        process_frame = frame.copy()
                        enhanced = self.enhance_document_image(process_frame)
                        scale_factor = 1.5
                        new_width = int(enhanced.shape[1] * scale_factor)
                        new_height = int(enhanced.shape[0] * scale_factor)
                        enlarged_enhanced = cv2.resize(enhanced, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
                        cv2.imshow("enhanced Document", enlarged_enhanced)
                        cv2.waitKey(1000)
                        cv2.destroyAllWindows()

                        processing_complete_flag['value'] = True
                        processing_thread = threading.Thread(
                            target=self.process_document,
                            args=(enhanced.copy(), processing_complete_flag)
                        )
                        processing_thread.start()
                        processing_thread_holder['thread'] = processing_thread
                    except Exception as e:
                        print(f"Error during manual processing: {e}")
                        self.speak_text("Error processing document", "en")
                elif key == ord('1'):  # '1' for translation (was 't')
                    self.speak_text("Translating Started", "en")
                    detected, translated = self.translator.translate_from_frame(frame)
                    print("Detected:", detected)
                    print("Translated:", translated)

              
                elif key == 27 or key == ord('0'):  # '0' for exit (was 'q')
                    print("Exit command received from OCR mode")
                    self.speak_text("Shutting down system", "en")
                    self.running = False
                    switch_mode = 'exit'
                    break
                    
        except Exception as e:
            print(f"Error in OCR processing: {e}")
            self.speak_text("Error in OCR processing", "en")
            return False
        finally:
            # --- EDITED: Add exit message ---
            self.speak_text("Exiting OCR system.")
            time.sleep(1) # Give the message time to play
            
            if processing_thread_holder['thread'] and processing_thread_holder['thread'].is_alive():
                processing_thread_holder['thread'].join()
            
            # --- EDITED: Release the camera here if it was opened by this method ---
            if self.cap:
                self.cap.release()
                self.cap = None
            cv2.destroyAllWindows()
            
            self.running = False
            
        return switch_mode


# def main_ocr():
#     """Main function to run the OCR system with proper camera handling"""
#     try:
#         # Initialize the camera ONCE
#         cap = cv2.VideoCapture(0)
        
#         # Initialize the OCR processor with the camera
#         ocr = OCRProcessor(cap=cap)
        
#         # Optimize camera settings
#         ocr.cap = ocr.optimize_camera_settings(ocr.cap)
        
#         # Display startup message
#         # print("="*50)
#         # print("OCR System Starting")
#         # print("="*50)
#         ocr.speak_text("OCR system ready", "en")
        
#         # Start the main OCR processing loop
#         switch_mode = ocr.capture_and_process()
        
#         # Handle mode switching (if implemented in your full application)
#         if switch_mode:
#             print(f"Switching to {switch_mode} mode...")
#             # Here you would call your mode switching logic
        
#     except Exception as e:
#         print(f"Critical error in OCR system: {e}")
#         import traceback
#         traceback.print_exc()
#     finally:
#         if cap is not None:
#             cap.release()
#         cv2.destroyAllWindows()
#         print("OCR system shutdown")


def main_ocr(ocr_processor=None):
    """Main function to run the OCR system with proper camera handling"""
    cap = None
    try:
        if ocr_processor is not None:
            # Use the preloaded OCRProcessor, just set up the camera
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow for faster camera init on Windows
            ocr_processor.cap = cap
            # Skip speaking "OCR system ready" as it's redundant
            switch_mode = ocr_processor.capture_and_process()
        else:
            # Fallback: create a new OCRProcessor
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow for faster camera init
            ocr = OCRProcessor(cap=cap)
            ocr.speak_text("OCR system ready", "en")
            switch_mode = ocr.capture_and_process()
    except Exception as e:
        print(f"Critical error in OCR system: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("OCR system shutdown")


if __name__ == "__main__":
    main_ocr()