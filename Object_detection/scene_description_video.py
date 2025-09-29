# --- START OF FILE scene_description_video.py ---

import cv2
import time
import threading
import logging
import os
import re
import uuid # For unique temporary filenames
from PIL import Image
import numpy as np
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions # For upload retry
from audio_feedback_vision_assitant import AudioFeedbackHandler # Import the shared handler

# Configure basic logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Video Scene Description Class ---

class VideoSceneDescriber:
    # Recommend gemini-1.5-flash for video tasks balancing cost/speed/capability
    def __init__(self, api_key, model_name="gemini-1.5-flash-latest"):
        """
        Initializes the Video Scene Description system using Gemini.

        Args:
            api_key (str): Google AI API key for Gemini.
            model_name (str): The Gemini model capable of video input.
        """
        self.api_key = api_key
        self.model_name = model_name
        genai.configure(api_key=self.api_key)

        # Safety settings (adjust as needed)
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        # Generation configuration
        self.generation_config = {
            "temperature": 0.4, # Slightly higher temp might be good for descriptive video summary
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 2048,
        }

        try:
            if 'gemini-1.5' not in self.model_name: # 1.5 models generally handle video
                 logger.warning(f"Model '{self.model_name}' might not be ideal for video. Consider 'gemini-1.5-flash-latest' or 'gemini-1.5-pro-latest'.")

            self.model = genai.GenerativeModel(
                self.model_name,
                safety_settings=safety_settings,
                generation_config=self.generation_config
            )
            logger.info(f"Video Scene Describer initialized with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model '{self.model_name}': {e}", exc_info=True)
            raise

    def _frame_to_pil_image(self, frame_bgr):
        """Convert OpenCV frame (BGR) to PIL Image (RGB)"""
        if frame_bgr is None: return None
        try:
            rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb_frame)
        except cv2.error as e:
            logger.error(f"OpenCV error converting frame to PIL: {e}")
            return None

    def _create_prompt(self):
        """Creates the structured prompt specifically for VIDEO analysis."""
        prompt = """Analyze this VIDEO clip (approximately 5-10 seconds) and provide a comprehensive description for a visually impaired person, focusing on actions and changes.
Structure your response MUST follow these sections:

1.  **SCENE OVERVIEW:**
    *   Brief summary (1-2 sentences) of the main activity or setting in the video.
    *   Environment type (e.g., indoor room type, outdoor setting).
    *   General lighting and atmosphere throughout the clip.

2.  **KEY OBJECTS & PEOPLE:**
    *   List the most prominent objects and people visible throughout the video.
    *   For STATIC key objects, describe their estimated location (e.g., 'table in the center', 'door on the left wall').
    *   For people or objects that MOVE SIGNIFICANTLY, describe their initial or most common location.

3.  **ACTIONS & EVENTS:**
    *   Describe the sequence of main actions or events occurring in the video (e.g., 'A person walks from left to right', 'The camera pans across the room', 'A cup is placed on the table').
    *   Note any significant interactions between objects or people.
    *   Mention any text visible for a duration, or sounds if clearly relevant (though focus on visuals).

4.  **SPATIAL & TEMPORAL SUMMARY:**
    *   Describe the overall spatial layout and how it might change due to camera or subject movement.
    *   Summarize the key movements relative to the scene or camera over time.

Focus on accuracy, clear descriptions of movement and interaction."""
        return prompt

    def analyze_video(self, uploaded_video_file, representative_frame_np):
        """
        Analyzes an uploaded video file using the Gemini API.

        Args:
            uploaded_video_file: File object from genai.upload_file().
            representative_frame_np (np.ndarray): A frame (BGR) from the video
                                                 to use for drawing labels.

        Returns:
            dict: Contains 'text_description', 'representative_image_pil' (PIL),
                  or 'error'.
        """
        if uploaded_video_file is None:
            return {"error": "No uploaded video file provided"}

        # Wait for file to be in ACTIVE state
        print(f"Checking file state for {uploaded_video_file.name}...")
        logger.info(f"Checking file state for {uploaded_video_file.name}...")
        max_wait_time = 120  # Maximum wait time in seconds
        check_interval = 2   # Check every 2 seconds
        elapsed_time = 0
        
        while elapsed_time < max_wait_time:
            try:
                # Get current file info
                file_info = genai.get_file(uploaded_video_file.name)
                print(f"File state: {file_info.state.name} (waiting {elapsed_time}s/{max_wait_time}s)")
                logger.debug(f"File state: {file_info.state.name}")
                
                if file_info.state.name == "ACTIVE":
                    print(f"✓ File {uploaded_video_file.name} is now ACTIVE and ready for analysis.")
                    logger.info(f"File {uploaded_video_file.name} is now ACTIVE and ready for analysis.")
                    break
                elif file_info.state.name == "FAILED":
                    print(f"✗ File {uploaded_video_file.name} failed to process on server.")
                    logger.error(f"File {uploaded_video_file.name} failed to process on server.")
                    return {"error": "File processing failed on server"}
                else:
                    print(f"⏳ File {uploaded_video_file.name} is in {file_info.state.name} state. Waiting...")
                    logger.info(f"File {uploaded_video_file.name} is in {file_info.state.name} state. Waiting...")
                    time.sleep(check_interval)
                    elapsed_time += check_interval
                    
            except Exception as e:
                print(f"⚠️ Error checking file state: {e}. Continuing with analysis attempt...")
                logger.warning(f"Error checking file state: {e}. Continuing with analysis attempt...")
                break
        
        if elapsed_time >= max_wait_time:
            print(f"⚠️ File {uploaded_video_file.name} did not become ACTIVE within {max_wait_time}s. Attempting analysis anyway...")
            logger.warning(f"File {uploaded_video_file.name} did not become ACTIVE within {max_wait_time}s. Attempting analysis anyway...")

        prompt = self._create_prompt()
        print(f"📤 Sending video analysis request to {self.model_name}...")
        logger.info(f"Sending video analysis request ({uploaded_video_file.name}) to {self.model_name}...")

        try:
            # Generate content using the video file object
            response = self.model.generate_content(
                [prompt, uploaded_video_file],
                request_options={"timeout": 600} # Increased timeout for video processing
            )
            print("✓ Received video analysis response from Gemini.")
            logger.info("Received video analysis response from Gemini.")

            try:
                if response.prompt_feedback.block_reason:
                    reason = response.prompt_feedback.block_reason
                    logger.error(f"Request blocked by API. Reason: {reason}")
                    for rating in response.prompt_feedback.safety_ratings: logger.error(f"  Safety Rating: {rating.category} - {rating.probability}")
                    return {"error": f"Blocked by API: {reason}"}
                full_text = response.text
            except (AttributeError, ValueError, IndexError) as e_resp:
                 logger.error(f"Error accessing Gemini response content: {e_resp}", exc_info=True)
                 logger.error(f"Prompt Feedback: {response.prompt_feedback if hasattr(response, 'prompt_feedback') else 'N/A'}")
                 return {"error": f"Invalid response structure from API: {e_resp}"}
            except Exception as e_gen:
                 logger.error(f"Unexpected error during Gemini content generation: {e_gen}", exc_info=True)
                 return {"error": f"Gemini generation error: {e_gen}"}

            logger.debug(f"--- START Gemini Video Full Text ---\n{full_text}\n--- END Gemini Video Full Text ---")

            text_description = {}
            sections_map = {
                "scene overview": "scene_overview",
                "key objects & people": "objects_people",
                "actions & events": "actions_events",
                "spatial & temporal summary": "spatial_temporal",
            }
            section_content = {key: [] for key in sections_map.values()}
            current_section_key = None

            logger.debug("--- Starting VIDEO Section Parsing ---")
            lines = full_text.splitlines()

            for line_num, line in enumerate(lines):
                stripped_line = line.strip()
                if not stripped_line:
                    if current_section_key:
                         section_content[current_section_key].append(line)
                    continue

                potential_header = re.sub(r'[\*#:\d\.]', '', stripped_line).strip().lower()

                matched_key = None
                for header_keyword, key in sections_map.items():
                    if potential_header.startswith(header_keyword):
                        logger.debug(f"Line {line_num+1}: Detected Header for VIDEO Section: '{key}' from line: '{stripped_line}'")
                        current_section_key = key
                        matched_key = key
                        break

                if not matched_key and current_section_key:
                    if not re.match(r'^[\*\-\•]\s*$', stripped_line):
                        section_content[current_section_key].append(stripped_line)

            for key, lines_list in section_content.items():
                joined_content = '\n'.join(lines_list).strip()
                if joined_content:
                    text_description[key] = joined_content
                    logger.debug(f"Assigned content for VIDEO key '{key}'. Length: {len(joined_content)}")
                else:
                    if lines_list: logger.warning(f"Empty content after processing for VIDEO section key '{key}'.")
                    else: logger.debug(f"No content lines found for VIDEO section key '{key}'.")

            text_description["full_text"] = full_text

            if not any(text_description.get(key) for key in sections_map.values()):
                 logger.error("Failed to parse any meaningful video content sections.")

            rep_image_pil = self._frame_to_pil_image(representative_frame_np)

            return {
                "text_description": text_description,
                "representative_image_pil": rep_image_pil
            }

        except google_exceptions.RetryError as e_retry:
             logger.error(f"Gemini API request failed after retries: {e_retry}", exc_info=True)
             return {"error": f"API Request Failed (RetryError): {e_retry}"}
        except Exception as e:
            logger.error(f"Error analyzing video with Gemini: {e}", exc_info=True)
            return {"error": f"Gemini API Error: {e}"}

    def create_speech_output(self, text_description):
        """Creates structured speech output from the video analysis result."""
        if not text_description or not isinstance(text_description, dict):
            return [{"text": "Error: Invalid text description format.", "section": "error"}]

        speech_segments = []
        sections_to_speak = [
            ("scene_overview", "Scene overview"),
            ("objects_people", "Key Objects and People"),
            ("actions_events", "Actions and Events"),
            ("spatial_temporal", "Spatial and Temporal Summary")
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

        if not speech_segments:
            full_text = text_description.get("full_text")
            if full_text:
                logger.warning("Video sections not parsed for speech, using cleaned full text.")
                text = full_text
                text = re.sub(r'\d+\.\s*\*\*[\w\s&]+:\*\*', '', text, flags=re.IGNORECASE)
                text = re.sub(r'^[\*\-\•]\s*', '', text, flags=re.MULTILINE)
                text = re.sub(r'\*+', '', text)
                text = re.sub(r'\s+', ' ', text).strip()
                if text: 
                    speech_segments.append({
                        "text": "General Video Summary: " + text,
                        "section": "general",
                        "section_name": "General Video Summary"
                    })
            else: 
                speech_segments.append({
                    "text": "No video description available to speak.",
                    "section": "error",
                    "section_name": "Error"
                })

        return speech_segments

def run_video_scene_description(api_key, audio_handler, duration=7, camera_id=2, save_output=True):
    """
    Runs the video scene description system.
    """
    print("🎬 Initializing Video Scene Description System...")
    logger.info("Initializing Video Scene Description System...")
    describer = None
    cap = None
    uploaded_file = None
    temp_video_filename = None
    user_interrupted_flow = False # Flag for early exit by user

    if not audio_handler:
         print("❌ AudioFeedbackHandler is required for video scene description.")
         logger.error("AudioFeedbackHandler is required for video scene description.")
         return

    try:
        audio_handler.speak("Welcome to video scene description mode.")
        print("🤖 Initializing Gemini model...")
        describer = VideoSceneDescriber(api_key=api_key)
        print("✓ Video Scene Describer initialized successfully.")

        def capture_and_analyze_video():
            nonlocal cap, uploaded_file, temp_video_filename, user_interrupted_flow
            
            print(f"📷 Attempting to open camera ID: {camera_id}")
            logger.info(f"Attempting to open camera ID: {camera_id}")
            cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)  # Use DirectShow backend for better compatibility on Windows
            if not cap.isOpened():
                if camera_id != 2:
                    print("⚠️ Main camera not responding. Trying backup camera...")
                    audio_handler.speak("Main camera not responding. Let me try another camera.")
                    cap.open(0)
                    if not cap.isOpened():
                        print("❌ Could not find any working cameras.")
                        audio_handler.speak("I couldn't find any working cameras. Please check your camera connection.")
                        return None
                    else:
                        print("✓ Backup camera connected successfully.")
                        audio_handler.speak("Backup camera connected successfully.")
                else:
                    print("❌ Camera not found. Please check your connection.")
                    audio_handler.speak("Camera not found. Please check your connection.")
                    return None

            w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_prop=cap.get(cv2.CAP_PROP_FPS); fps=fps_prop if fps_prop > 0 else 30.0
            if w==0 or h==0: 
                print("❌ Failed to get valid frame dimensions.")
                logger.error("Failed to get valid frame dimensions.")
                audio_handler.speak("Error getting camera dimensions.")
                return None
            
            print(f"✓ Camera opened: {w}x{h} @ {fps:.2f} FPS. Target duration: {duration}s")
            logger.info(f"Camera opened: {w}x{h} @ {fps:.2f} FPS. Target duration: {duration}s")

            print(f"🎥 Starting video capture for {duration} seconds...")
            logger.info(f"Capturing video for ~{duration} seconds...")
            frames = []
            start_time = time.time()
            capture_window_name = "Capturing Video for Description..."

            while time.time() - start_time < duration:
                ret, frame = cap.read()
                if not ret: 
                    logger.warning("Could not read frame.")
                    time.sleep(0.1)
                    continue
                frames.append(frame.copy())
                display_frame = frame.copy()
                elapsed = int(time.time() - start_time)
                cv2.putText(display_frame, f"REC {elapsed}/{duration}s (Press '0')", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow(capture_window_name, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('0'): 
                    print("⏹️ Capture interrupted by user.")
                    logger.info("Capture interrupted.")
                    if audio_handler.speaking: audio_handler.force_stop()
                    user_interrupted_flow = True
                    break

            try: cv2.destroyWindow(capture_window_name)
            except cv2.error: logger.debug("Capture window already closed.")
            if cap.isOpened(): cap.release(); logger.info("Camera released after capture.")

            if user_interrupted_flow:
                return None

            if not frames:
                print("❌ No video frames captured.")
                audio_handler.speak("I couldn't record any video frames. Please try again.")
                return None
            
            print(f"✓ Captured {len(frames)} frames successfully.")
            logger.info(f"Captured {len(frames)} frames.")
            middle_frame_np = frames[len(frames) // 2].copy()

            temp_video_filename = f"temp_video_{uuid.uuid4()}.mp4"
            print(f"💾 Encoding video to temporary file: {temp_video_filename}")
            logger.info(f"Encoding video to temporary file: {temp_video_filename}")
            video_writer = None
            try:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(temp_video_filename, fourcc, fps, (w, h))
                if not video_writer.isOpened(): raise IOError("Could not open video writer.")
                for frame in frames: video_writer.write(frame)
                print("✓ Temporary video saved successfully.")
                logger.info("Temporary video saved successfully.")
            except Exception as e:
                print(f"❌ Error writing temporary video file: {e}")
                logger.error(f"Error writing temporary video file: {e}", exc_info=True)
                if os.path.exists(temp_video_filename): os.remove(temp_video_filename)
                audio_handler.speak("Error saving video.")
                return None
            finally:
                if video_writer: video_writer.release()
            del frames

            audio_handler.speak("Video recorded successfully. Now uploading it for analysis. This might take a moment.")
            print(f"☁️ Uploading video to Google Cloud: {temp_video_filename}")
            logger.info(f"Uploading video: {temp_video_filename}...")
            upload_start_time = time.time()
            uploaded_file = None
            try:
                for attempt in range(3):
                    try:
                        print(f"📤 Upload attempt {attempt + 1}/3...")
                        uploaded_file = genai.upload_file(path=temp_video_filename, display_name="Scene Video")
                        upload_time = time.time() - upload_start_time
                        print(f"✓ Video uploaded successfully: '{uploaded_file.name}' (Upload took {upload_time:.2f}s)")
                        logger.info(f"Video uploaded: '{uploaded_file.name}'. URI: '{uploaded_file.uri}' (Upload took {upload_time:.2f}s)")
                        break
                    except google_exceptions.RetryError as e_retry:
                        print(f"⚠️ Upload attempt {attempt+1} failed (RetryError). Retrying in {2**(attempt+1)}s...")
                        logger.warning(f"Upload attempt {attempt+1} failed (RetryError): {e_retry}. Retrying in {2**(attempt+1)}s...")
                        if attempt == 2: raise
                        time.sleep(2**(attempt+1))
                    except Exception as e_upload:
                        print(f"⚠️ Upload attempt {attempt+1} failed: {e_upload}")
                        logger.error(f"Upload attempt {attempt+1} failed: {e_upload}", exc_info=True)
                        if attempt == 2: raise
                        time.sleep(2**(attempt+1))

                if uploaded_file is None: raise Exception("Video upload failed after retries.")

            except Exception as e:
                print(f"❌ Video upload process failed: {e}")
                logger.error(f"Video upload process failed: {e}", exc_info=True)
                if os.path.exists(temp_video_filename): os.remove(temp_video_filename)
                audio_handler.speak("I had trouble uploading the video. Please try again.")
                return None

            audio_handler.speak("Upload complete. The video is now being processed. This may take a minute.")
            print("🧠 Analyzing uploaded video with Gemini AI...")
            logger.info("Analyzing uploaded video with Gemini (this may take time)...")
            processing_frame = middle_frame_np.copy()
            cv2.putText(processing_frame, "Processing Video on Server...", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,165,255), 2)
            cv2.putText(processing_frame, "Please wait...", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)
            processing_window_name = "Processing Video..."
            cv2.imshow(processing_window_name, processing_frame); cv2.waitKey(1)

            analysis_result = describer.analyze_video(uploaded_file, middle_frame_np)

            try: cv2.destroyWindow(processing_window_name)
            except cv2.error: logger.debug("Processing window already closed.")

            return analysis_result, middle_frame_np

        # Initial capture and analysis
        print("🎬 Starting initial video capture and analysis...")
        result = capture_and_analyze_video()
        if result is None or user_interrupted_flow:
            if user_interrupted_flow:
                print("⏹️ Video capture cancelled by user.")
                audio_handler.speak("Video capture cancelled.")
                time.sleep(1.5)
            return
        
        analysis_result, middle_frame_np = result

        if "error" in analysis_result:
            error_msg = analysis_result['error']
            print(f"❌ Video analysis failed: {error_msg}")
            logger.error(f"Video analysis failed: {error_msg}")
            audio_handler.speak(f"I'm sorry, but I had trouble understanding the video. {error_msg}")
            error_frame = middle_frame_np.copy()
            cv2.putText(error_frame, "ERROR:",(20,80),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
            y0, dy = 110, 20
            for i, line in enumerate(error_msg.split('\n')):
                if y0+i*dy > middle_frame_np.shape[0]-20: break
                cv2.putText(error_frame,line,(20,y0+i*dy),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
            cv2.imshow("Analysis Error", error_frame); cv2.waitKey(0)
            return

        print("✅ Analysis complete! Processing results...")
        audio_handler.speak("Analysis complete. I'll now describe what I saw in the video.")
        text_description = analysis_result.get("text_description")
        representative_pil_image = analysis_result.get("representative_image_pil")

        def display_and_print_results(text_desc, is_retake=False):
            """Helper function to display and print results consistently"""
            if text_desc:
                prefix = "RETAKE - " if is_retake else ""
                print(f"\n--- {prefix}Gemini VIDEO Description ---")
                for key, title in [("scene_overview", "SCENE OVERVIEW"), ("objects_people", "KEY OBJECTS & PEOPLE"),
                                  ("actions_events", "ACTIONS & EVENTS"), ("spatial_temporal", "SPATIAL & TEMPORAL SUMMARY")]:
                    print(f"\n{title}:")
                    content = text_desc.get(key, "N/A")
                    print(content)
                print("\n-------------------------------\n")
            else: 
                print(f"\n{prefix}No text description parsed.\n")

        display_and_print_results(text_description)

        result_window_name = "Scene Description Result (Video) - Press '0' to Quit or 'Enter' for Retake"
        if middle_frame_np is not None:
            cv2.imshow(result_window_name, middle_frame_np)
            logger.info("Displaying representative frame.")
        else:
            logger.warning("Could not display representative frame.")

        def speak_description_segments(text_desc, segments_desc=""):
            """Helper function to speak description segments consistently"""
            if text_desc:
                speech_segments = describer.create_speech_output(text_desc)
                print(f"🔊 Speaking video description{segments_desc}...")
                logger.info("Speaking video description...")
                audio_handler.speak("Here's what I found in the video:")
                
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
                logger.warning("No description generated to speak.")
                audio_handler.speak("No video description was generated.")
                return False

        # Speak initial description
        interrupted = speak_description_segments(text_description)
        if interrupted:
            user_interrupted_flow = True
            audio_handler.speak("Description stopped.")
            time.sleep(1.5)
        else:
            audio_handler.speak("Video description complete.")

        def save_results(text_desc, frame_np, is_retake=False):
            """Helper function to save results consistently"""
            # Save results even if speech was interrupted, but not if capture was interrupted
            if save_output and text_desc and text_desc.get("full_text"):
                save_msg = "Saving the video analysis results" + (" (retake)" if is_retake else "") + " for later reference."
                print(f"💾 {save_msg}")
                if audio_handler and not audio_handler.speaking:  # Only speak if not currently speaking
                    audio_handler.speak("Saving analysis results.")
                
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                # Create organized folder structure
                main_output_dir = "saved_outputs"
                video_output_dir = os.path.join(main_output_dir, "scene_outputs_video")
                os.makedirs(video_output_dir, exist_ok=True)
                
                retake_suffix = "_retake" if is_retake else ""
                base_filename = os.path.join(video_output_dir, f"scene_video_{timestamp}{retake_suffix}")

                if text_desc and text_desc.get("full_text"):
                    txt_file = f"{base_filename}_description.txt"
                    try:
                        with open(txt_file, 'w', encoding='utf-8') as f:
                            f.write(f"Model: {describer.model_name}\nTimestamp: {timestamp}\nVideo Duration: ~{duration}s\n")
                            if is_retake:
                                f.write("Type: Retake\n")
                            f.write("Status: Complete Analysis\n")
                            f.write("\n")
                            f.write(text_desc["full_text"])
                        print(f"✓ Text description saved to: {txt_file}")
                        logger.info(f"Text description saved to {txt_file}")
                    except Exception as e: 
                        print(f"❌ Failed to save text: {e}")
                        logger.error(f"Failed to save text: {e}")

                if frame_np is not None:
                    img_file = f"{base_filename}_frame.png"
                    try:
                        cv2.imwrite(img_file, frame_np)
                        print(f"✓ Representative frame saved to: {img_file}")
                        logger.info(f"Representative frame saved to {img_file}")
                    except Exception as e: 
                        print(f"❌ Failed to save frame: {e}")
                        logger.error(f"Failed to save frame: {e}")

        # Save initial results - always save if analysis was successful
        save_results(text_description, middle_frame_np)

        if not user_interrupted_flow:
            print("✅ Video description process complete.")
            audio_handler.speak("Video description process complete. Press 0 to exit or Enter to record a new video.")
        
        # Main interaction loop
        print("⌨️ Waiting for user input: 0 (quit) or Enter (retake)")
        while not user_interrupted_flow:
            window_exists = False
            try:
                if cv2.getWindowProperty(result_window_name, cv2.WND_PROP_VISIBLE) >= 0:
                    window_exists = True
            except cv2.error:
                window_exists = False

            if not window_exists:
                print("🪟 Result window closed by user.")
                logger.info("Result window no longer exists. Exiting wait loop.")
                if audio_handler.speaking: audio_handler.force_stop()
                user_interrupted_flow = True
                break

            key = cv2.waitKey(100) & 0xFF
            if key == ord('0'):
                print("⏹️ '0' key pressed. Exiting application.")
                logger.info("'0' key pressed. Exiting.")
                if audio_handler.speaking: audio_handler.force_stop()
                user_interrupted_flow = True
                break
            elif key == 13:  # Enter key
                print("🔄 'Enter' key pressed. Starting video retake...")
                logger.info("'Enter' key pressed. Recording new video.")
                if audio_handler.speaking: 
                    audio_handler.force_stop()
                    time.sleep(0.2)
                
                # Clean up previous files before retake
                if temp_video_filename and os.path.exists(temp_video_filename):
                    try:
                        os.remove(temp_video_filename)
                        print(f"🗑️ Deleted previous temp video: {temp_video_filename}")
                        logger.info(f"Deleted previous temp video: {temp_video_filename}")
                    except Exception as e:
                        print(f"⚠️ Error deleting previous temp video: {e}")
                        logger.error(f"Error deleting previous temp video: {e}")
                
                if uploaded_file is not None:
                    try:
                        genai.delete_file(uploaded_file.name)
                        print(f"🗑️ Deleted previous uploaded file: {uploaded_file.name}")
                        logger.info(f"Deleted previous uploaded file: {uploaded_file.name}")
                    except Exception as e:
                        print(f"⚠️ Error deleting previous uploaded file: {e}")
                        logger.error(f"Error deleting previous uploaded file: {e}")
                
                audio_handler.speak("Recording a new video.")
                cv2.destroyAllWindows()
                time.sleep(0.5)
                
                # Reset variables
                uploaded_file = None
                temp_video_filename = None
                
                # Capture and analyze new video
                print("🎬 Starting retake video capture and analysis...")
                result = capture_and_analyze_video()
                if result is None or user_interrupted_flow:
                    if not user_interrupted_flow:
                        print("❌ Failed to record new video.")
                        audio_handler.speak("Failed to record new video.")
                    break
                
                analysis_result, middle_frame_np = result
                
                if "error" in analysis_result:
                    error_msg = analysis_result['error']
                    print(f"❌ Retake video analysis failed: {error_msg}")
                    logger.error(f"Video analysis failed: {error_msg}")
                    audio_handler.speak(f"I'm sorry, but I had trouble understanding the video. {error_msg}")
                    break
                
                print("✅ Retake analysis complete! Processing results...")
                audio_handler.speak("Analysis complete. I'll now describe what I saw in the new video.")
                text_description = analysis_result.get("text_description")
                
                # Display and print retake results
                display_and_print_results(text_description, is_retake=True)
                
                if middle_frame_np is not None:
                    cv2.imshow(result_window_name, middle_frame_np)
                
                # Save retake results BEFORE speaking (so it's saved even if speech is interrupted)
                save_results(text_description, middle_frame_np, is_retake=True)
                
                # Speak retake description
                interrupted = speak_description_segments(text_description, " (retake)")
                if interrupted:
                    # Don't set user_interrupted_flow here since we already saved
                    print("⏹️ Speech interrupted but results are saved.")
                    audio_handler.speak("Description stopped, but results have been saved.")
                    time.sleep(1.5)
                    user_interrupted_flow = True  # Now set it to exit
                    break
                else:
                    audio_handler.speak("Video description complete.")
                
                if not user_interrupted_flow:
                    print("✅ Retake video description process complete.")
                    audio_handler.speak("Video description process complete. Press 0 to exit or Enter to record a new video.")

    except Exception as e:
        print(f"💥 Critical error occurred: {e}")
        logger.critical(f"An unhandled error occurred: {e}", exc_info=True)
        if audio_handler:
            if audio_handler.speaking: audio_handler.force_stop(); time.sleep(0.2)
            audio_handler.speak("I encountered a problem and need to restart. Please try again.")
        user_interrupted_flow = True
    finally:
        if audio_handler:
            print("👋 Shutting down video scene description...")
            audio_handler.speak("Thank you for using video scene description. Goodbye!")
        logger.info("Shutting down Video Scene Description...")
        if cap and cap.isOpened(): cap.release(); logger.info("Camera released (final check).")
        cv2.destroyAllWindows(); logger.info("OpenCV windows closed.")

        # Clean up temporary files
        if temp_video_filename and os.path.exists(temp_video_filename):
            try:
                os.remove(temp_video_filename)
                print(f"🗑️ Deleted temporary video file: {temp_video_filename}")
                logger.info(f"Deleted temporary video file: {temp_video_filename}")
            except Exception as e:
                print(f"⚠️ Error deleting temp video {temp_video_filename}: {e}")
                logger.error(f"Error deleting temp video {temp_video_filename}: {e}")

        # Clean up uploaded files from cloud
        if uploaded_file is not None:
            print(f"🗑️ Cleaning up uploaded file from cloud: {uploaded_file.name}")
            logger.info(f"Attempting to delete uploaded file from cloud: {uploaded_file.name}")
            try:
                # Add a small delay before attempting deletion
                time.sleep(1)
                genai.delete_file(uploaded_file.name)
                print(f"✓ Successfully deleted uploaded file: {uploaded_file.name}")
                logger.info(f"Successfully deleted uploaded file: {uploaded_file.name}")
            except Exception as e:
                print(f"⚠️ Could not delete uploaded file '{uploaded_file.name}' from cloud: {e}")
                logger.warning(f"Could not delete uploaded file '{uploaded_file.name}' from cloud: {e}")
                # Don't treat this as a critical error since it's cleanup
        
        print("🏁 Video scene description shutdown complete.")
        logger.info("Shutdown complete.")

# --- Example Usage ---
if __name__ == "__main__":
    print("Running Video Scene Description Standalone Test...")
    api_key = os.environ.get("GOOGLE_API_KEY")

    if not api_key:
        api_key = "Your_api_key"
        if api_key == "YOUR_API_KEY_HERE":
            print("\nERROR: GOOGLE_API_KEY environment variable not set, and placeholder not replaced.")
            print("Please set the environment variable or replace the placeholder in the code.")
            exit()
        else:
            print("WARNING: Using hardcoded API key from script. Use environment variable for security.")

    test_audio_handler = None
    try:
        test_audio_handler = AudioFeedbackHandler()
        run_video_scene_description(
            api_key=api_key,
            audio_handler=test_audio_handler,
            duration=7,
            camera_id=1,
            save_output=True
        )
    except Exception as e:
         print(f"Standalone test failed: {e}")
    finally:
        if test_audio_handler:
            test_audio_handler.stop()
        cv2.destroyAllWindows()
        print("Standalone test finished.")

