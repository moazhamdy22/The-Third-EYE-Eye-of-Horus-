# --- START OF NEW FILE Face_recognation/main_face.py ---

import logging
import sys
import os

# Add the current directory to the Python path to ensure local imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Now import the Face Recognition System
try:
    from Face_Recognition_System import UnifiedFaceRecognition
except ImportError:
    # Fallback for different import scenarios
    import Face_Recognition_System
    UnifiedFaceRecognition = Face_Recognition_System.UnifiedFaceRecognition

logger = logging.getLogger(__name__)

def main():
    """
    Entry point for the Face Recognition system.
    No audio handler is used in this version.
    """
    logger.info("--- Entering Face Recognition Module ---")
    try:
        # Initialize the face recognition system without audio handler
        face_system = UnifiedFaceRecognition()
        face_system.run_Face_Recognition()
    except Exception as e:
        logger.error(f"An error occurred in the Face Recognition module: {e}", exc_info=True)
    finally:
        logger.info("--- Exiting Face Recognition Module ---")

if __name__ == "__main__":
    # This allows the module to be tested standalone
    print("Running Face Recognition Module in standalone mode...")
    logging.basicConfig(level=logging.INFO)
    main()