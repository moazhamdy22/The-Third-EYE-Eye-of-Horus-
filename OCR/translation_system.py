import cv2
import logging
import os
import uuid
import google.generativeai as genai
from deep_translator import GoogleTranslator
from PIL import Image
from langdetect import detect, LangDetectException
from gtts import gTTS
import pygame

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FrameTranslator:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash-latest')
        self.generation_config = {
            "temperature": 0.3,
            "top_p": 1,
            "top_k": 32,
            "max_output_tokens": 4096,
        }

    def detect_text(self, image_path):
        try:
            img = Image.open(image_path)
            response = self.model.generate_content(
                ["Extract all visible text EXACTLY as it appears in this image. Return only the text.", img],
                generation_config=self.generation_config
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return None

    def capture_and_translate(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to capture frame from camera.")
            cap.release()
            return None, None

        # Show the captured frame
        cv2.imshow("Captured Frame", frame)
        cv2.waitKey(2000)  # Show for 2 seconds
        cv2.destroyAllWindows()

        # Save frame temporarily
        temp_img = f"temp_{uuid.uuid4()}.jpg"
        cv2.imwrite(temp_img, frame)

        detected_text = None
        translated = None

        try:
            detected_text = self.detect_text(temp_img)
            os.remove(temp_img)
            if detected_text and detected_text.strip():
                try:
                    detected_lang = detect(detected_text)
                    target_lang = 'en' if detected_lang == 'ar' else 'ar'
                    translated = GoogleTranslator(source='auto', target=target_lang).translate(detected_text)
                    logger.info(f"Extracted Text: {detected_text}")
                    logger.info(f"Translated Text: {translated}")

                    # Play sound for extracted text
                    self.play_sound_feedback(detected_text, detected_lang)
                    # Play sound for translated text
                    self.play_sound_feedback(translated, target_lang)

                except LangDetectException as e:
                    logger.error(f"Language detection failed: {e}")
                except Exception as e:
                    logger.error(f"Translation error: {e}")
            else:
                logger.info("No text detected")
        except Exception as e:
            logger.error(f"Processing failed: {e}")

        cap.release()
        cv2.destroyAllWindows()
        return detected_text, translated

    def translate_from_frame(self, frame):
        # for the integration 
        """
        Accepts a BGR frame (numpy array), saves it temporarily, and performs detection and translation.
        """
        import cv2, uuid, os
        detected_text = None
        translated = None
        temp_img = f"temp_{uuid.uuid4()}.jpg"
        try:
            cv2.imwrite(temp_img, frame)
            detected_text = self.detect_text(temp_img)
            if detected_text and detected_text.strip():
                from langdetect import detect, LangDetectException
                from deep_translator import GoogleTranslator
                detected_lang = detect(detected_text)
                target_lang = 'en' if detected_lang == 'ar' else 'ar'
                translated = GoogleTranslator(source='auto', target=target_lang).translate(detected_text)
                # Play sound for extracted text
                self.play_sound_feedback(detected_text, detected_lang)
                # Play sound for translated text
                self.play_sound_feedback(translated, target_lang)
        except Exception as e:
            logger.error(f"Translation from frame failed: {e}")
        finally:
            if os.path.exists(temp_img):
                os.remove(temp_img)
        return detected_text, translated

    @staticmethod
    def play_sound_feedback(text, lang):
        """
        Generate and play sound feedback for the given text and language using pygame.
        """
        try:
            tts = gTTS(text=text, lang=lang)
            audio_file = f"tts_{uuid.uuid4()}.mp3"
            tts.save(audio_file)
            pygame.mixer.init()
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            pygame.mixer.music.unload()
            os.remove(audio_file)
        except Exception as e:
            logger.error(f"TTS error: {e}")

def main(api_key):
    translator = FrameTranslator(api_key)
    detected_text, translated = translator.capture_and_translate()
    if detected_text and translated:
        print(f"Extracted Text: {detected_text}")
        print(f"Translated Text: {translated}")

if __name__ == "__main__":
    api_key = "Your_api_key"  # Replace with your API key
    main(api_key)