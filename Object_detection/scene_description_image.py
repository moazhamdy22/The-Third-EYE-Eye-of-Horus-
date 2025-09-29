# --- START OF FILE scene_description_image.py ---

import cv2
import numpy as np
import time
import threading
import logging
import os
import re
import sys
from PIL import Image
import google.generativeai as genai

# Add current directory to Python path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from audio_feedback_vision_assitant import AudioFeedbackHandler # Import the shared handler

# Configure basic logging - REMOVE this. Configured in main.py
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Frame Selection Utilities (from object_detection_capture.py) ---
def calculate_blur_score(frame):
    """Calculates Laplacian variance as a measure of blur."""
    if frame is None: return 0
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = np.var(laplacian)
        return variance
    except cv2.error as e:
        logger.warning(f"OpenCV error calculating blur score: {e}")
        return 0

def select_best_frame(frames, blur_threshold=50.0):
    """Selects the least blurry frame based on Laplacian variance."""
    if not frames:
        logger.warning("No frames provided to select_best_frame.")
        return None

    best_frame_index = -1
    max_variance = -1.0
    sharp_frames_indices = []

    # logger.info(f"Calculating blur scores for {len(frames)} captured frames...") # Less verbose
    variances = []
    for i, frame_item in enumerate(frames): # Renamed frame to frame_item
        variance = calculate_blur_score(frame_item)
        variances.append(variance)
        if variance > max_variance:
            max_variance = variance
            best_frame_index = i
        if variance >= blur_threshold:
            sharp_frames_indices.append(i)
        # logger.debug(f"Frame {i} blur score: {variance:.2f}") # Too verbose for general use

    if not sharp_frames_indices:
        logger.warning(f"All frames seem blurry (max variance {max_variance:.2f} < threshold {blur_threshold}). Using frame with highest variance (Index {best_frame_index}).")
        if best_frame_index == -1:
             # logger.error("Could not determine best frame index even as fallback.") # Redundant if frames exist
             return frames[len(frames) // 2] if frames else None
        return frames[best_frame_index]
    elif len(sharp_frames_indices) == 1:
         selected_index = sharp_frames_indices[0]
         # logger.info(f"Selected the only sharp frame (Index {selected_index}) with score {variances[selected_index]:.2f}.")
         return frames[selected_index]
    else:
        middle_sharp_idx_in_list = len(sharp_frames_indices) // 2
        original_frame_index = sharp_frames_indices[middle_sharp_idx_in_list]
        # logger.info(f"Multiple sharp frames found. Selected middle sharp frame (Index {original_frame_index}) with score {variances[original_frame_index]:.2f}.")
        return frames[original_frame_index]
# --- End Frame Selection Utilities ---

# --- Main Scene Description Class ---

class ImageSceneDescriber:
    def __init__(self, api_key, model_name="gemini-1.5-flash-latest"):
        self.api_key = api_key
        self.model_name = model_name
        genai.configure(api_key=self.api_key)
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        self.generation_config = {
            "temperature": 0.4, "top_p": 0.95, "top_k": 40, "max_output_tokens": 2048,
        }
        try:
            self.model = genai.GenerativeModel(
                self.model_name,
                safety_settings=safety_settings,
                generation_config=self.generation_config
            )
            # logger.info(f"Image Scene Describer initialized with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model '{self.model_name}': {e}", exc_info=True)
            raise

    def _frame_to_pil_image(self, frame_bgr):
        if frame_bgr is None: return None
        try:
            rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb_frame)
        except cv2.error as e:
            logger.error(f"OpenCV error converting frame to PIL: {e}")
            return None

    def _create_prompt(self):
        prompt = """Analyze this image and provide a comprehensive description of the scene for a visually impaired person.
Structure your response MUST follow these sections:
1.  **SCENE OVERVIEW:** (Summary, environment, lighting)
2.  **OBJECTS & LAYOUT:** (Prominent objects, name, location, relative position)
3.  **DETAILED OBSERVATIONS:** (Object details, text, people general, noteworthy items)
4.  **ACCESSIBILITY & SPATIAL SUMMARY:** (Closest objects, overall arrangement)
Focus on accuracy and clear descriptions useful for navigation and understanding."""
        return prompt # Simplified prompt for brevity in example

    def analyze_image(self, frame_bgr):
        if frame_bgr is None: return {"error": "No frame provided"}
        pil_image = self._frame_to_pil_image(frame_bgr)
        if pil_image is None: return {"error": "Frame conversion failed"}
        prompt_text = self._create_prompt() # Corrected variable name
        try:
            print(f"📤 Sending image analysis request to {self.model_name}...")
            # logger.info(f"Sending image analysis request to {self.model_name}...")
            response = self.model.generate_content([prompt_text, pil_image]) # Use prompt_text
            print("✓ Received response from Gemini.")
            # logger.info("Received response from Gemini.")
            if response.prompt_feedback.block_reason:
                reason = response.prompt_feedback.block_reason
                print(f"❌ Request blocked by API. Reason: {reason}")
                logger.error(f"Request blocked by API. Reason: {reason}")
                return {"error": f"Blocked by API: {reason}"}
            full_text = response.text
        except Exception as e:
            print(f"❌ Error analyzing image with Gemini: {e}")
            logger.error(f"Error analyzing image with Gemini: {e}", exc_info=True)
            return {"error": f"Gemini API Error: {e}"}

        # logger.debug(f"--- START Gemini Full Text Response ---\n{full_text}\n--- END Gemini Full Text Response ---")
        text_description = {}
        sections_map = {
            "scene overview": "scene_overview", "objects & layout": "objects_layout",
            "detailed observations": "detailed_observations", "accessibility & spatial summary": "accessibility_spatial",
        }
        section_content = {key: [] for key in sections_map.values()}
        current_section_key = None
        lines = full_text.splitlines()
        for line in lines:
            stripped_line = line.strip()
            if not stripped_line:
                if current_section_key: section_content[current_section_key].append(line)
                continue
            potential_header = re.sub(r'[\*#:\d\.]', '', stripped_line).strip().lower()
            matched_key = None
            for header_keyword, key in sections_map.items():
                if potential_header.startswith(header_keyword):
                    current_section_key = key
                    matched_key = key
                    break
            if not matched_key and current_section_key:
                if not re.match(r'^[\*\-\•]\s*$', stripped_line):
                    section_content[current_section_key].append(stripped_line)
        for key, lines_list in section_content.items():
            joined_content = '\n'.join(lines_list).strip()
            if joined_content: text_description[key] = joined_content
        text_description["full_text"] = full_text
        return {"text_description": text_description, "analyzed_image_pil": pil_image}

    def create_speech_output(self, text_description):
        if not text_description or not isinstance(text_description, dict):
            return [{"text": "Error: Invalid text description format.", "section": "error"}]
        
        speech_segments = []
        sections_to_speak = [
            ("scene_overview", "Scene overview"), 
            ("objects_layout", "Objects and Layout"),
            ("detailed_observations", "Detailed Observations"), 
            ("accessibility_spatial", "Accessibility and Spatial Summary")
        ]
        
        for key, prefix in sections_to_speak:
            if text_description.get(key):
                text = text_description[key]
                text = re.sub(r'^[\*\-\•]\s*', '', text, flags=re.MULTILINE)
                text = re.sub(r'\*+', '', text)
                text = re.sub(r'\s+', ' ', text).strip()
                if text: 
                    speech_segments.append({
                        "text": f"{prefix}: {text}",
                        "section": key,
                        "section_name": prefix
                    })
        
        if not speech_segments and text_description.get("full_text"): # Fallback
            logger.warning("Sections not parsed for speech, using cleaned full text.")
            text = text_description["full_text"]
            # Basic cleaning for full text fallback
            text = re.sub(r'\d+\.\s*\*\*[\w\s&]+:\*\*', '', text, flags=re.IGNORECASE) # Remove "1. **HEADER:**"
            text = re.sub(r'^[\*\-\•]\s*', '', text, flags=re.MULTILINE)
            text = re.sub(r'\*+', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            if text: 
                speech_segments.append({
                    "text": "General Scene Summary: " + text,
                    "section": "general",
                    "section_name": "General Scene Summary"
                })
        
        return speech_segments
# --- End create_speech_output ---

# --- Main Execution Function ---
def run_image_scene_description(api_key, audio_handler, duration=5, camera_id=1, blur_thresh=60.0, save_output=True):
    print("📸 Initializing Image Scene Description System...")
    logger.info("Initializing Image Scene Description System...")
    describer = None
    cap = None
    user_interrupted_flow = False # Flag for early exit by user

    if not audio_handler:
         print("❌ AudioFeedbackHandler is required for scene description.")
         logger.error("AudioFeedbackHandler is required for scene description.")
         return

    try:
        audio_handler.speak("Welcome to scene description mode.")
        print("🤖 Initializing Gemini model...")
        describer = ImageSceneDescriber(api_key=api_key)
        print("✓ Image Scene Describer initialized successfully.")

        print(f"📷 Attempting to open camera ID: {camera_id}")
        # logger.info(f"Attempting to open camera ID: {camera_id}")
        cap = cv2.VideoCapture(camera_id,cv2.CAP_DSHOW) # Use DirectShow for Windows compatibility
        if not cap.isOpened():
            if camera_id != 2:
                print("⚠️ Main camera not responding. Trying backup camera...")
                audio_handler.speak("Main camera not responding. Trying backup camera.")
                cap.open(0)
                if not cap.isOpened():
                    print("❌ Could not find any working cameras.")
                    audio_handler.speak("Could not find any working cameras.")
                    return
                else:
                    print("✓ Backup camera connected.")
                    audio_handler.speak("Backup camera connected.")
            else: # camera_id was 0 and failed
                print("❌ Camera not found. Please check connection.")
                audio_handler.speak("Camera not found. Please check connection.")
                return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"✓ Camera opened: {w}x{h}")
        # logger.info(f"Camera opened: {w}x{h}")

        # --- Capture Phase ---
        print(f"📸 Starting image capture for {duration} seconds...")
        # logger.info(f"Capturing frames for {duration} seconds...")
        frames = []
        start_time = time.time()
        capture_window_name = "Capturing Scene for Description..."
        audio_handler.speak("Capturing scene. Please hold steady.")

        while time.time() - start_time < duration:
            ret, frame_cap = cap.read() # Renamed frame to frame_cap
            if not ret:
                # logger.warning("Could not read frame during capture.")
                time.sleep(0.1)
                if not cap.isOpened(): # Check if camera disconnected
                    logger.error("Camera disconnected during capture.")
                    audio_handler.speak("Error: Camera disconnected.")
                    user_interrupted_flow = True
                    break
                continue
            frames.append(frame_cap.copy())
            display_frame = frame_cap.copy()
            elapsed = int(time.time() - start_time)
            cv2.putText(display_frame, f"Capturing... {elapsed}/{duration}s (Press '0')", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow(capture_window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('0'):
                logger.info("Capture interrupted by user.")
                if audio_handler.speaking: audio_handler.force_stop()
                user_interrupted_flow = True
                break
        
        if user_interrupted_flow: # Early exit from capture
            try: cv2.destroyWindow(capture_window_name)
            except cv2.error: pass
            if cap and cap.isOpened(): cap.release()
            audio_handler.speak("Scene capture cancelled.")
            time.sleep(1.5)
            return

        try: cv2.destroyWindow(capture_window_name)
        except cv2.error: pass # logger.debug("Capture window already closed.")
        if cap.isOpened(): cap.release() # logger.info("Camera released after capture.")

        if not frames:
            print("❌ No frames captured.")
            # logger.error("No frames captured.")
            audio_handler.speak("Failed to capture any images.")
            return

        print("🔍 Selecting the best frame from captured images...")
        audio_handler.speak("Images captured. Selecting the best one.")
        best_frame = select_best_frame(frames, blur_threshold=blur_thresh)
        del frames 
        if best_frame is None:
            print("❌ Failed to select a clear frame.")
            # logger.error("Failed to select a best frame.")
            audio_handler.speak("Could not select a clear frame. Please try again.")
            return

        # --- Analysis Phase ---
        print("🧠 Analyzing the scene with Gemini AI...")
        audio_handler.speak("Analyzing the scene. This might take a moment.")
        # logger.info("Analyzing selected frame with Gemini...")
        processing_frame_display = best_frame.copy() # Use a different variable for display
        cv2.putText(processing_frame_display, "Analyzing with Gemini...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
        cv2.imshow("Processing Image...", processing_frame_display)
        cv2.waitKey(1) # Allow window to render. Gemini call is blocking.

        analysis_result = describer.analyze_image(best_frame)

        try: cv2.destroyWindow("Processing Image...")
        except cv2.error: pass # logger.debug("Processing window already closed.")

        if "error" in analysis_result:
            error_msg = analysis_result['error']
            print(f"❌ Scene analysis failed: {error_msg}")
            logger.error(f"Scene analysis failed: {error_msg}")
            audio_handler.speak(f"Sorry, I had trouble analyzing the scene. {error_msg}")
            # ... (error display logic remains the same) ...
            error_frame_display = best_frame.copy()
            cv2.putText(error_frame_display, "ERROR:", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y0, dy = 110, 20
            for i, line in enumerate(error_msg.split('\n')):
                if y0 + i * dy > h - 20: break
                cv2.putText(error_frame_display, line, (20, y0 + i * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imshow("Analysis Error", error_frame_display); cv2.waitKey(0) # Wait for key press on error
            user_interrupted_flow = True # Treat as an interruption for exit messaging
            return

        print("✅ Analysis complete! Processing results...")
        audio_handler.speak("Analysis complete.")
        text_description = analysis_result.get("text_description")
        
        def display_and_print_results(text_desc, is_retake=False):
            """Helper function to display and print results consistently"""
            if text_desc:
                prefix = "RETAKE - " if is_retake else ""
                print(f"\n--- {prefix}Gemini IMAGE Description ---")
                for key, title in [("scene_overview", "SCENE OVERVIEW"), ("objects_layout", "OBJECTS & LAYOUT"),
                                  ("detailed_observations", "DETAILED OBSERVATIONS"), ("accessibility_spatial", "ACCESSIBILITY & SPATIAL SUMMARY")]:
                    print(f"\n{title}:")
                    content = text_desc.get(key, "N/A")
                    print(content)
                print("\n-------------------------------\n")
            else: 
                print(f"\n{prefix}No text description parsed.\n")

        display_and_print_results(text_description)

        result_window_name = "Scene Description Result - Press '0' to Quit or 'Enter' for Next"
        if best_frame is not None:
            cv2.imshow(result_window_name, best_frame)
        
        def speak_description_segments(text_desc, segments_desc=""):
            """Helper function to speak description segments consistently"""
            if text_desc:
                print(f"🔊 Speaking image description{segments_desc}...")
                audio_handler.speak("Here's what I found in the scene:")
                speech_segments = describer.create_speech_output(text_desc)
                
                for i, segment in enumerate(speech_segments):
                    # Check if user already interrupted before starting this segment
                    if user_interrupted_flow:
                        break
                        
                    print(f"🗣️ Speaking section {i+1}/{len(speech_segments)}: {segment['section_name']}")
                    # Speak the current segment
                    audio_handler.speak(segment["text"])
                    
                    # Wait for speech to complete or user interruption
                    while audio_handler.speaking and not user_interrupted_flow:
                        key = cv2.waitKey(50) & 0xFF
                        window_visible = True
                        try:
                            if cv2.getWindowProperty(result_window_name, cv2.WND_PROP_VISIBLE) < 1:
                                window_visible = False
                        except cv2.error: 
                            window_visible = False

                        if key == ord('0') or not window_visible:
                            print("⏹️ User interrupted description speech.")
                            logger.info("User interrupted description speech.")
                            audio_handler.force_stop()
                            return True  # Return True if interrupted
                    
                    # If user interrupted during speech, don't continue to next section
                    if user_interrupted_flow:
                        break
                    
                    # Check if this is the last segment
                    if i < len(speech_segments) - 1:  # Not the last segment
                        audio_handler.speak(f"Press Enter to continue to the next section: {speech_segments[i+1]['section_name']}, or 0 to quit.")
                        print(f"⏳ Waiting for user input: Enter (next) or 0 (quit)")
                        
                        # Wait for user input
                        waiting_for_input = True
                        while waiting_for_input and not user_interrupted_flow:
                            key = cv2.waitKey(100) & 0xFF
                            window_visible = True
                            try:
                                if cv2.getWindowProperty(result_window_name, cv2.WND_PROP_VISIBLE) < 1:
                                    window_visible = False
                            except cv2.error: 
                                window_visible = False

                            if key == 13:  # Enter key
                                waiting_for_input = False
                                print("▶️ User chose to continue to next section.")
                                logger.info("User chose to continue to next section.")
                            elif key == ord('0') or not window_visible:
                                print("⏹️ User chose to quit during section break.")
                                logger.info("User chose to quit during section break.")
                                if audio_handler.speaking: 
                                    audio_handler.force_stop()
                                return True  # Return True if interrupted
                
                return False  # Return False if completed normally
            else:
                print("⚠️ No description generated to speak.")
                audio_handler.speak("I couldn't generate a description for this scene.")
                return False

        # Speak initial description
        interrupted = speak_description_segments(text_description)
        if interrupted:
            user_interrupted_flow = True
            audio_handler.speak("Description stopped.")
            time.sleep(1.5)
        else:
            audio_handler.speak("Scene description complete.")

        def save_results(text_desc, frame, is_retake=False):
            """Helper function to save results consistently"""
            # Save results even if speech was interrupted, but not if capture was interrupted
            if save_output and text_desc and text_desc.get("full_text"):
                save_msg = "Saving the results" + (" (retake)" if is_retake else "") + " for later reference."
                print(f"💾 {save_msg}")
                if audio_handler and not audio_handler.speaking:  # Only speak if not currently speaking
                    audio_handler.speak("Saving analysis results.")
                
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                # Create organized folder structure
                main_output_dir = "saved_outputs"
                image_output_dir = os.path.join(main_output_dir, "scene_outputs_image")
                os.makedirs(image_output_dir, exist_ok=True)
                
                retake_suffix = "_retake" if is_retake else ""
                base_filename = os.path.join(image_output_dir, f"scene_{timestamp}{retake_suffix}")
                full_text_content = text_desc.get("full_text") if text_desc else None
                if full_text_content:
                    txt_file = f"{base_filename}_description.txt"
                    try:
                        with open(txt_file, 'w', encoding='utf-8') as f:
                            f.write(f"Model: {describer.model_name}\nTimestamp: {timestamp}\n")
                            if is_retake:
                                f.write("Type: Retake\n")
                            f.write("Status: Complete Analysis\n")
                            f.write(f"\n{full_text_content}")
                        print(f"✓ Text description saved to: {txt_file}")
                        # logger.info(f"Text description saved to {txt_file}")
                    except Exception as e: 
                        print(f"❌ Failed to save text: {e}")
                        logger.error(f"Failed to save text: {e}")
                if frame is not None:
                    img_file = f"{base_filename}_original.png"
                    try: 
                        cv2.imwrite(img_file, frame)
                        print(f"✓ Image saved to: {img_file}")
                        # logger.info(f"Image saved to {img_file}")
                    except Exception as e: 
                        print(f"❌ Failed to save image: {e}")
                        logger.error(f"Failed to save image: {e}")

        # Save initial results - always save if analysis was successful
        save_results(text_description, best_frame)

        # Speak initial description
        interrupted = speak_description_segments(text_description)
        if interrupted:
            # Don't set user_interrupted_flow here since we already saved
            print("⏹️ Speech interrupted but results are saved.")
            audio_handler.speak("Description stopped, but results have been saved.")
            time.sleep(1.5)
            user_interrupted_flow = True
        else:
            audio_handler.speak("Scene description complete.")

        # --- Final Wait & Cleanup ---
        if not user_interrupted_flow: # If flow was not interrupted before this
            print("✅ Image description process complete.")
            audio_handler.speak("Description process complete. Press 0 to exit or Enter to take a new image.")
        
        print("⌨️ Waiting for user input: 0 (quit) or Enter (retake)")
        while not user_interrupted_flow: # Loop only if not already interrupted
            window_exists_and_visible = False
            try:
                if cv2.getWindowProperty(result_window_name, cv2.WND_PROP_VISIBLE) >= 1: # Check for visibility
                    window_exists_and_visible = True
            except cv2.error: pass # Window doesn't exist or error checking

            if not window_exists_and_visible: # If window closed by 'X'
                print("🪟 Result window closed by user.")
                logger.info("Result window closed by user (final wait).")
                if audio_handler.speaking: audio_handler.force_stop()
                user_interrupted_flow = True # Set flag to ensure proper exit message
                break

            key = cv2.waitKey(100) & 0xFF
            if key == ord('0'):
                print("⏹️ '0' key pressed. Exiting application.")
                logger.info("'0' key pressed (final wait).")
                if audio_handler.speaking: audio_handler.force_stop()
                user_interrupted_flow = True # Set flag
                break
            
            if key == 13:  # Enter key
                print("🔄 'Enter' key pressed. Starting image retake...")
                if audio_handler.speaking: 
                    audio_handler.force_stop()
                    time.sleep(0.2)
                audio_handler.speak("Taking a new image.")
                cv2.destroyAllWindows()
                time.sleep(0.5)
                
                # Restart from capture phase without re-initializing describer
                print("📷 Reopening camera for retake...")
                cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW) # Use DirectShow for Windows compatibility
                if not cap.isOpened():
                    print("❌ Error: Camera not found for retake.")
                    audio_handler.speak("Error: Camera not found.")
                    return
                
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                # --- New Capture Phase ---
                print(f"📸 Starting retake capture for {duration} seconds...")
                frames = []
                start_time = time.time()
                capture_window_name = "Capturing Scene for Description..."
                audio_handler.speak("Capturing scene. Please hold steady.")

                while time.time() - start_time < duration:
                    ret, frame_cap = cap.read()
                    if not ret:
                        time.sleep(0.1)
                        continue
                    frames.append(frame_cap.copy())

                    display_frame = frame_cap.copy()
                    elapsed = int(time.time() - start_time)
                    cv2.putText(display_frame, f"Capturing... {elapsed}/{duration}s (Press '0')", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow(capture_window_name, display_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('0'):
                        audio_handler.force_stop()
                        user_interrupted_flow = True
                        break

                if user_interrupted_flow:
                    try: cv2.destroyWindow(capture_window_name)
                    except cv2.error: pass
                    if cap and cap.isOpened(): cap.release()
                    return
                
                try: cv2.destroyWindow(capture_window_name)
                except cv2.error: pass
                
                if cap and cap.isOpened(): cap.release()
                
                if not frames:
                    audio_handler.speak("Failed to capture any images.")
                    return
                
                # Process new frames
                print("🔍 Selecting the best frame from retake images...")
                audio_handler.speak("Images captured. Selecting the best one.")
                best_frame = select_best_frame(frames, blur_threshold=blur_thresh)
                del frames
                
                if best_frame is None:
                    print("❌ Could not select a clear frame from retake.")
                    audio_handler.speak("Could not select a clear frame. Please try again.")
                    return
                
                # Analyze new frame
                print("🧠 Analyzing the retake scene with Gemini AI...")
                audio_handler.speak("Analyzing the scene. This might take a moment.")
                processing_frame_display = best_frame.copy()
                cv2.putText(processing_frame_display, "Analyzing with Gemini...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                cv2.imshow("Processing Image...", processing_frame_display)
                cv2.waitKey(1)
                
                analysis_result = describer.analyze_image(best_frame)
                
                try: cv2.destroyWindow("Processing Image...")
                except cv2.error: pass
                
                if "error" in analysis_result:
                    error_msg = analysis_result['error']
                    print(f"❌ Retake scene analysis failed: {error_msg}")
                    logger.error(f"Scene analysis failed: {error_msg}")
                    audio_handler.speak(f"Sorry, I had trouble analyzing the scene. {error_msg}")
                    return
                
                print("✅ Retake analysis complete! Processing results...")
                audio_handler.speak("Analysis complete.")
                text_description = analysis_result.get("text_description")
                
                # Display and print retake results
                display_and_print_results(text_description, is_retake=True)
                
                if best_frame is not None:
                    cv2.imshow(result_window_name, best_frame)
                
                # Speak retake description
                interrupted = speak_description_segments(text_description, " (retake)")
                if interrupted:
                    # Don't set user_interrupted_flow here since we already saved
                    print("⏹️ Speech interrupted but results are saved.")
                    audio_handler.speak("Description stopped, but results have been saved.")
                    time.sleep(1.5)
                    user_interrupted_flow = True
                    return
                else:
                    audio_handler.speak("Scene description complete.")
                
                # Save retake results
                save_results(text_description, best_frame, is_retake=True)
                
                # Continue with the wait loop for the new analysis
                print("✅ Retake image description process complete.")
                audio_handler.speak("Description process complete. Press 0 to exit or Enter to take a new image.")
                continue

        # Final module-specific exit message handled by the finally block or if interrupted

    except Exception as e:
        print(f"💥 Critical error occurred in Image Scene Description: {e}")
        logger.critical(f"An unhandled error occurred in Image Scene Description: {e}", exc_info=True)
        if audio_handler:
            if audio_handler.speaking: audio_handler.force_stop(); time.sleep(0.2)
            audio_handler.speak("I encountered a problem and need to stop.")
            time.sleep(1.5)
        user_interrupted_flow = True # Ensure finally block knows it was an error exit

    finally:
        if user_interrupted_flow and audio_handler:
             # If interrupted by user or error, a more specific message might have already been spoken
             # or an error message. A generic "module shutting down" is good.
            pass # Specific messages handled within try block
        
        if audio_handler:
            print("👋 Shutting down image scene description...")
            audio_handler.speak("Image scene description module shutting down.")
            # time.sleep(1.0) # Give it a moment
            
        # logger.info("Shutting down Image Scene Description...")
        if cap and cap.isOpened():
            cap.release()
            # logger.info("Camera released.")
        cv2.destroyAllWindows()
        print("🏁 Image scene description shutdown complete.")
        # logger.info("Image Scene Description shutdown complete.")

# --- Example Usage (Standalone Test Block) ---
if __name__ == "__main__":
    # Standalone logging (remove basicConfig from top of file)
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Running Image Scene Description Standalone Test...") # Use configured logger

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key: # Fallback for testing if env var not set
        api_key = "your_api" # Replace with your actual key for testing
        if "YOUR_GOOGLE_API_KEY_HERE" in api_key: # Check if placeholder is still there
            logger.error("\nERROR: GOOGLE_API_KEY not set and placeholder not replaced.")
            exit()
        else:
            logger.warning("Using hardcoded API key for testing. Use environment variables for security.")

    test_audio_handler = None
    try:
        test_audio_handler = AudioFeedbackHandler()
        if not test_audio_handler.engine:
            logger.critical("Standalone test: AudioFeedbackHandler TTS engine failed. Exiting.")
        else:
            run_image_scene_description(
                api_key=api_key,
                audio_handler=test_audio_handler,
                duration=5, # Short duration for testing
                camera_id=1,
                blur_thresh=60.0,
                save_output=True
            )
    except Exception as e:
        logger.error(f"Standalone test failed: {e}", exc_info=True)
    finally:
        if test_audio_handler:
            logger.info("Stopping standalone test audio handler...")
            test_audio_handler.stop()
            logger.info("Standalone test audio handler stopped.")
        cv2.destroyAllWindows()
        logger.info("Standalone test finished.")