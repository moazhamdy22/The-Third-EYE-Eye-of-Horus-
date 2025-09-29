# --- START OF FILE scene_description_blip.py ---

import cv2
import numpy as np
import time
import logging
import os
import sys
import psutil
from PIL import Image

try:
    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Add current directory to Python path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from audio_feedback_vision_assitant import AudioFeedbackHandler

# --- Frame Selection Utilities (Copied from other modules for consistency) ---
from scene_description_image import calculate_blur_score, select_best_frame

logger = logging.getLogger(__name__)

def detect_and_select_device():
    """
    Automatically detects the best available device (GPU/CPU) and returns device info.
    
    Returns:
        tuple: (device_string, device_info_dict)
    """
    device_info = {
        'device_type': None,
        'device_name': None,
        'cuda_available': False,
        'gpu_count': 0,
        'gpu_memory_total': 0,
        'gpu_memory_free': 0,
        'cpu_count': psutil.cpu_count(),
        'ram_total': psutil.virtual_memory().total / (1024**3),  # GB
        'ram_available': psutil.virtual_memory().available / (1024**3)  # GB
    }
    
    if torch.cuda.is_available():
        device_info['cuda_available'] = True
        device_info['gpu_count'] = torch.cuda.device_count()
        device_info['device_name'] = torch.cuda.get_device_name(0)
        
        # Get GPU memory info
        gpu_memory = torch.cuda.get_device_properties(0)
        device_info['gpu_memory_total'] = gpu_memory.total_memory / (1024**3)  # GB
        
        torch.cuda.empty_cache()
        device_info['gpu_memory_free'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        
        device_string = "cuda"
        device_info['device_type'] = "GPU"
        
        logger.info(f"🚀 GPU DETECTED: {device_info['device_name']}")
        logger.info(f"📊 GPU Memory: {device_info['gpu_memory_total']:.2f} GB total")
        logger.info(f"⚡ Using GPU acceleration for BLIP model")
        
    else:
        device_string = "cpu"
        device_info['device_type'] = "CPU"
        device_info['device_name'] = f"{psutil.cpu_count()} cores"
        
        logger.info(f"💻 No GPU detected, using CPU: {device_info['device_name']}")
        logger.info(f"📊 System RAM: {device_info['ram_total']:.2f} GB total, {device_info['ram_available']:.2f} GB available")
        logger.warning("⚠️  CPU inference will be slower than GPU")
    
    return device_string, device_info

def log_system_stats(device_info, phase=""):
    """Log current system resource usage"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    logger.info(f"📈 {phase} System Stats:")
    logger.info(f"   CPU Usage: {cpu_percent:.1f}%")
    logger.info(f"   RAM Usage: {memory.percent:.1f}% ({memory.used / (1024**3):.2f}/{memory.total / (1024**3):.2f} GB)")
    
    if device_info['cuda_available'] and torch.cuda.is_available():
        gpu_memory_used = torch.cuda.memory_allocated(0) / (1024**3)
        gpu_memory_cached = torch.cuda.memory_reserved(0) / (1024**3)
        logger.info(f"   GPU Memory Used: {gpu_memory_used:.2f} GB")
        logger.info(f"   GPU Memory Cached: {gpu_memory_cached:.2f} GB")

class BlipSceneDescriber:
    """
    A class to generate image captions using the Salesforce BLIP model.
    """
    def __init__(self, model_path="Salesforce/blip-image-captioning-large", device=None):
        """
        Initializes the BLIP model and processor with automatic device detection.

        Args:
            model_path (str): The Hugging Face model path.
            device (str): The device to run the model on. If None, auto-detects best device.
        """
        if not TRANSFORMERS_AVAILABLE:
            logger.critical("Transformers or PyTorch library not found. BLIP describer cannot function.")
            raise ImportError("Please install PyTorch and Transformers: pip install torch transformers")

        # Auto-detect device if not specified
        if device is None:
            self.device, self.device_info = detect_and_select_device()
        else:
            self.device = device
            _, self.device_info = detect_and_select_device()
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                self.device = "cpu"
        
        self.model_path = model_path
        self.load_times = {}
        
        logger.info(f"🔧 Initializing BLIP model '{self.model_path}' on {self.device_info['device_type']}")
        log_system_stats(self.device_info, "PRE-LOAD")
        
        try:
            # Time the processor loading
            start_time = time.time()
            self.processor = BlipProcessor.from_pretrained(self.model_path)
            self.load_times['processor'] = time.time() - start_time
            
            # Time the model loading
            start_time = time.time()
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = BlipForConditionalGeneration.from_pretrained(
                    self.model_path, torch_dtype=torch.float16
                ).to(self.device)
                logger.info("✅ BLIP model loaded on GPU with float16 precision.")
            else:
                self.model = BlipForConditionalGeneration.from_pretrained(self.model_path).to(self.device)
                logger.info("✅ BLIP model loaded on CPU with float32 precision.")
            
            self.load_times['model'] = time.time() - start_time
            self.load_times['total'] = self.load_times['processor'] + self.load_times['model']
            
            # Log loading performance
            logger.info(f"⏱️  Loading Performance:")
            logger.info(f"   Processor: {self.load_times['processor']:.2f}s")
            logger.info(f"   Model: {self.load_times['model']:.2f}s")
            logger.info(f"   Total: {self.load_times['total']:.2f}s")
            
            log_system_stats(self.device_info, "POST-LOAD")
                
        except Exception as e:
            logger.error(f"❌ Failed to load BLIP model: {e}", exc_info=True)
            raise

    def _frame_to_pil_image(self, frame_bgr):
        """Converts an OpenCV BGR frame to a PIL RGB Image."""
        if frame_bgr is None: return None
        try:
            rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb_frame)
        except cv2.error as e:
            logger.error(f"OpenCV error converting frame to PIL: {e}")
            return None

    def analyze_image(self, frame_bgr):
        """
        Generates a caption for a single image frame with performance monitoring.

        Args:
            frame_bgr (numpy.ndarray): The image frame in BGR format.

        Returns:
            dict: A dictionary containing either the 'caption' or an 'error', plus timing info.
        """
        if frame_bgr is None:
            return {"error": "No frame provided for analysis."}
        
        analysis_start = time.time()
        
        # Log pre-inference stats
        log_system_stats(self.device_info, "PRE-INFERENCE")
        
        pil_image = self._frame_to_pil_image(frame_bgr)
        if pil_image is None:
            return {"error": "Failed to convert frame to a usable image format."}

        try:
            # Time preprocessing
            preprocess_start = time.time()
            inputs = self.processor(pil_image, return_tensors="pt").to(
                self.device, 
                torch.float16 if self.device == "cuda" else torch.float32
            )
            preprocess_time = time.time() - preprocess_start
            
            # Time inference
            inference_start = time.time()
            with torch.no_grad():  # Disable gradient computation for faster inference
                out = self.model.generate(**inputs, max_new_tokens=50)
            inference_time = time.time() - inference_start
            
            # Time postprocessing
            postprocess_start = time.time()
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            postprocess_time = time.time() - postprocess_start
            
            total_time = time.time() - analysis_start
            
            # Log performance metrics
            logger.info(f"⚡ Inference Performance on {self.device_info['device_type']}:")
            logger.info(f"   Preprocessing: {preprocess_time:.3f}s")
            logger.info(f"   Inference: {inference_time:.3f}s")
            logger.info(f"   Postprocessing: {postprocess_time:.3f}s")
            logger.info(f"   Total Analysis: {total_time:.3f}s")
            logger.info(f"🎯 Generated Caption: '{caption}'")
            
            # Log post-inference stats
            log_system_stats(self.device_info, "POST-INFERENCE")
            
            return {
                "caption": caption.strip(),
                "performance": {
                    "device": self.device_info['device_type'],
                    "device_name": self.device_info['device_name'],
                    "preprocess_time": preprocess_time,
                    "inference_time": inference_time,
                    "postprocess_time": postprocess_time,
                    "total_time": total_time
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error during BLIP model inference: {e}", exc_info=True)
            return {"error": f"Model inference failed: {e}"}

    def create_speech_output(self, analysis_result):
        """
        Formats the single-sentence caption for speech output.
        """
        if "error" in analysis_result:
            return [{"text": f"Error during analysis: {analysis_result['error']}", "section": "error"}]
        
        caption = analysis_result.get("caption")
        if not caption:
            return [{"text": "I could not generate a caption for the image.", "section": "no_caption"}]
            
        # The structure is a list of dicts to be consistent with other modules
        return [{
            "text": f"The scene can be described as: {caption}",
            "section": "blip_caption"
        }]

def run_blip_scene_description(
    audio_handler: AudioFeedbackHandler,
    duration=5,
    camera_id=1,
    model_path="Salesforce/blip-image-captioning-large",
    device=None,  # Changed default to None for auto-detection
    blur_thresh=60.0
):
    """
    Runs the full capture-then-analyze pipeline for BLIP image captioning with auto device detection.
    """
    logger.info("🚀 Initializing BLIP Scene Captioning System...")
    describer = None
    cap = None
    user_interrupted_flow = False

    if not audio_handler:
        logger.error("❌ A valid AudioFeedbackHandler is required for BLIP captioning.")
        return

    try:
        audio_handler.speak("Welcome to the quick caption mode.")
        
        # Initialize with auto device detection
        system_start = time.time()
        describer = BlipSceneDescriber(model_path=model_path, device=device)
        system_init_time = time.time() - system_start
        
        logger.info(f"🎯 System initialized in {system_init_time:.2f}s using {describer.device_info['device_type']}")
        
        # --- Retake Loop ---
        while True:
            # --- Capture Phase ---
            cap = cv2.VideoCapture(camera_id,cv2.CAP_DSHOW)
            if not cap.isOpened():
                audio_handler.speak("Error: Camera not found.")
                return

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            frames = []
            start_time = time.time()
            capture_window_name = "Capturing Scene for Captioning..."
            audio_handler.speak("Capturing scene. Please hold the camera steady.")

            while time.time() - start_time < duration:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.1)
                    if not cap.isOpened():
                        audio_handler.speak("Camera disconnected during capture.")
                        user_interrupted_flow = True
                        break
                    continue
                frames.append(frame.copy())
                
                display_frame = frame.copy()
                elapsed = int(time.time() - start_time)
                cv2.putText(display_frame, f"Capturing... {elapsed}/{duration}s (Press '0' to stop)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow(capture_window_name, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('0'):
                    audio_handler.force_stop()
                    user_interrupted_flow = True
                    break
            
            if cap.isOpened():
                cap.release()
            try:
                cv2.destroyWindow(capture_window_name)
            except cv2.error:
                pass

            if user_interrupted_flow:
                audio_handler.speak("Capture cancelled.")
                time.sleep(1.5)
                break 

            if not frames:
                audio_handler.speak("No images were captured. Please try again.")
                break

            # --- Analysis Phase with performance logging ---
            audio_handler.speak(f"Image captured. Analyzing for quick caption.")
            best_frame = select_best_frame(frames, blur_threshold=blur_thresh)
            del frames
            
            if best_frame is None:
                audio_handler.speak("Could not find a clear enough image. Please try again.")
                continue # Allows for retake

            processing_frame = best_frame.copy()
            cv2.putText(processing_frame, f"Generating Caption on {describer.device_info['device_type']}...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            cv2.imshow("Processing...", processing_frame)
            cv2.waitKey(1)
            
            analysis_result = describer.analyze_image(best_frame)
            
            try:
                cv2.destroyWindow("Processing...")
            except cv2.error:
                pass

            # --- Results Phase with performance info ---
            result_frame = best_frame.copy()
            result_window_name = "BLIP Caption Result - Press '0' to Exit, 'Enter' to Retake"
            
            if "error" in analysis_result:
                caption_text = f"Error: {analysis_result['error']}"
                audio_handler.speak(f"Sorry, I had a problem: {analysis_result['error']}")
            else:
                caption_text = analysis_result.get("caption", "No caption generated.")
                
                # Add performance info to speech if available
                if "performance" in analysis_result:
                    perf = analysis_result["performance"]
                    perf_text = f"Caption generated in {perf['total_time']:.1f} seconds using {perf['device']}."
                    logger.info(f"📊 {perf_text}")
                
                speech_segments = describer.create_speech_output(analysis_result)
                for segment in speech_segments:
                    audio_handler.speak(segment['text'])

            # Display caption and device info on the image
            cv2.rectangle(result_frame, (0, result_frame.shape[0] - 80), (result_frame.shape[1], result_frame.shape[0]), (0,0,0), -1)
            cv2.putText(result_frame, caption_text, (10, result_frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Add device info
            device_text = f"Device: {describer.device_info['device_type']} ({describer.device_info['device_name']})"
            cv2.putText(result_frame, device_text, (10, result_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            
            cv2.imshow(result_window_name, result_frame)
            
            audio_handler.speak("Analysis complete. Press 0 to exit, or Enter to take a new image.")
            
            # --- Wait for user action ---
            while True:
                key = cv2.waitKey(100) & 0xFF
                window_closed = False
                try:
                    if cv2.getWindowProperty(result_window_name, cv2.WND_PROP_VISIBLE) < 1:
                        window_closed = True
                except cv2.error:
                    window_closed = True
                
                if window_closed or key == ord('0'):
                    user_interrupted_flow = True
                    break
                
                if key == 13:  # Enter key for retake
                    logger.info("User requested retake.")
                    audio_handler.speak("Taking a new image.")
                    cv2.destroyWindow(result_window_name)
                    time.sleep(0.5)
                    break # Breaks inner while, continues outer while for retake
            
            if user_interrupted_flow:
                break # Breaks outer while loop to exit function

    except Exception as e:
        logger.critical(f"❌ A critical error occurred in BLIP captioning: {e}", exc_info=True)
        if audio_handler:
            audio_handler.speak("A critical error occurred. I need to stop.")
            time.sleep(1.5)
    finally:
        logger.info("🔄 Shutting down BLIP captioning system.")
        if audio_handler and not audio_handler.speaking:
            audio_handler.speak("Exiting quick caption mode.")
        if cap and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    if not TRANSFORMERS_AVAILABLE:
        logger.error("Cannot run standalone test: PyTorch or Transformers not installed.")
    else:
        logger.info("🚀 Starting BLIP Scene Captioning Standalone Test...")
        test_audio_handler = None
        try:
            test_audio_handler = AudioFeedbackHandler()
            if not test_audio_handler.engine:
                logger.error("AudioFeedbackHandler failed to initialize.")
            else:
                run_blip_scene_description(
                    audio_handler=test_audio_handler,
                    duration=4,
                    camera_id=1,
                    model_path="Salesforce/blip-image-captioning-large",
                    device=None,  # Auto-detect best device
                    blur_thresh=60.0
                )
        except Exception as e:
            logger.error(f"Standalone test failed: {e}", exc_info=True)
        finally:
            if test_audio_handler:
                test_audio_handler.stop()
            cv2.destroyAllWindows()