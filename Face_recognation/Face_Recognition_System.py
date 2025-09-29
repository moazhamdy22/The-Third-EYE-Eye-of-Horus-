import cv2
import numpy as np
import tensorflow as tf
from keras_facenet import FaceNet
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.metrics import confusion_matrix

from sklearn.svm import SVC
from mtcnn.mtcnn import MTCNN
from PIL import Image
import math
import os
import joblib
from sklearn.metrics import accuracy_score
import torch
import pyttsx3
import pygame
from gtts import gTTS
import time
from speech_to_text import SpeechToText

from audio_feedback_vision_assitant import AudioFeedbackHandler


class UnifiedFaceRecognition:
    def __init__(self, camera_index=1, shared_audio_handler=None):
        # --- System Configuration ---
        self.camera_index = camera_index
        self.recognition_interval = 1.0  # Time in seconds between recognition attempts
        self.dataset_folder = "dataset"
        self.PROBABILITY_THRESHOLD = 70  # 75% confidence threshold
        
        # Initialize speech to text with fallback to CPU
        try:
            print("Attempting to initialize speech recognition with CUDA...")
            self.speech_to_text = SpeechToText(model_size="medium", device="cuda", compute_type="int8")
            print("Successfully initialized speech recognition with CUDA")
        except RuntimeError as e:
            if "CUDA" in str(e):
                print(f"CUDA initialization failed: {e}")
                print("Falling back to CPU for speech recognition...")
                try:
                    self.speech_to_text = SpeechToText(model_size="base", device="cpu", compute_type="int8")
                    print("Successfully initialized speech recognition with CPU")
                except Exception as cpu_error:
                    print(f"CPU fallback also failed: {cpu_error}")
                    print("Speech recognition will be disabled")
                    self.speech_to_text = None
            else:
                print(f"Speech recognition initialization failed: {e}")
                self.speech_to_text = None
        except Exception as e:
            print(f"Unexpected error initializing speech recognition: {e}")
            self.speech_to_text = None
        
        # --- System State ---
        self.camera = None
        self.exit_flag = False
        self.processing_frame = False
        self.processing = False  # Add this flag to track active processing
        self.display_results = False
        self.current_results = []
        self.database_modified = False
        self.last_recognition_time = 0
     
        
        # --- EDITED: Store the shared audio handler ---
        self.audio_handler = shared_audio_handler
        
        # --- EDITED: Remove the old redundant TTS engine ---
        # self.tts_engine = pyttsx3.init()
        
        # --- Create Required Directories ---
        self.setup_directories()
        
        # --- GPU Setup ---
        self.setup_gpu()
        
        # --- Load Models ---
        print("Initializing face detection and recognition models...")
        self.detector = MTCNN()
        self.facenet_model = FaceNet()
        self.in_encoder = Normalizer(norm='l2')
        self.out_encoder = LabelEncoder()
        
        # --- Load Existing Model ---
        self.embeddings = None
        self.names = None
        self.encodings = None
        self.face_paths = None
        self.model = None
        self.load_model()
    
    def setup_directories(self):
        """Create necessary directories for the system"""
        os.makedirs(os.path.join(self.dataset_folder, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.dataset_folder, 'test'), exist_ok=True)
    
    def setup_gpu(self):
        """Configure GPU settings for optimal performance"""
        self.using_gpu = torch.cuda.is_available()
        print("PyTorch GPU Available:", self.using_gpu)
        print("Using device:", "cuda" if self.using_gpu else "cpu")
        
        # Configure TensorFlow GPU settings
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(f"Found {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
            except RuntimeError as e:
                print(e)
        
        # Check OpenCV CUDA support
        self.opencv_cuda_enabled = False
        try:
            cv2.cuda.setDevice(0)
            self.opencv_cuda_enabled = True
            print("OpenCV CUDA support enabled")
        except:
            print("OpenCV CUDA support not available")
    
    # --- Voice Feedback Functions ---
    
    def speak(self, text):
        """Use the shared audio handler to speak messages."""
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

    def speak_gtts(self, text):
        """Convert text to speech using gTTS and play it without opening a window"""
        if not text:
            return

        try:
            audio_file = "speech_output.mp3"
            tts = gTTS(text=text, lang="en")
            tts.save(audio_file)

            # Initialize pygame mixer
            pygame.mixer.init()
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()

            # Wait until playback finishes
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

            # Stop and unload the file before deleting
            pygame.mixer.music.stop()
            pygame.mixer.quit()
            os.remove(audio_file)  # Delete the file safely

        except Exception as e:
            print(f"Error in TTS: {e}")
    
    def speak_results(self, results):
        """Use Google TTS to announce recognition results"""
        if not results:
            self.speak("No faces detected.")
            return

        # Extract names and count known and unknown faces
        known_faces = [r['name'] for r in results if r['name'].lower() != "unknown"]
        unknown_faces = [r['name'] for r in results if r['name'].lower() == "unknown"]

        known_count = len(known_faces)
        unknown_count = len(unknown_faces)

         # Print confidences for all faces
        print("\nDetailed Recognition Results:")
        for result in results:
            name = result['name']
            prob = result.get('probability', 0)
            print(f"- {name}: {prob:.1f}% confidence")

        count_announcement = f"Detected {known_count} known faces and {unknown_count} unknown faces."
        print(count_announcement)
        self.speak(count_announcement)


        # Announce known face names
        if known_count > 0:
            if known_count == 1:
                names = known_faces[0]  # Only one name
            elif known_count == 2:
                names = " and ".join(known_faces)  # Two names
            else:
                names = ", ".join(known_faces[:-1]) + f", and {known_faces[-1]}"  # More than two names

            self.speak_gtts(f"The known faces are: {names}.")
    
    def speak_database_info(self, info):
        """Speak the database information"""
        self.speak(f"Database contains {info.get('total_people', 0)} people with {info.get('total_faces', 0)} total faces")
    
    def speak_face_locations(self, results, frame_width):
        """Announce the location of each recognized person."""
        positions = {"left": [], "right": [], "front": []}
        for r in results:
            if r['name'].lower() == "unknown":
                continue
            pos = self.get_horizontal_position(r['bbox'], frame_width)
            positions[pos].append(r['name'])
        messages = []
        for pos in ["left", "front", "right"]:
            if positions[pos]:
                names = " and ".join(positions[pos])
                messages.append(f"{names} on the {pos}")
        if messages:
            self.speak_gtts(", ".join(messages) + ".")
        else:
            self.speak("No known faces detected.")
    
    # --- Model Management Functions ---
    
    def load_model(self):
        """Load pre-trained model and embeddings"""
        try:
            data = np.load('embeddings.npz')
            # Check if the file has the expected arrays
            if 'arr_0' in data and 'arr_1' in data and 'arr_2' in data and 'arr_3' in data:
                self.embeddings = data['arr_0']  # Face embeddings
                self.names = data['arr_1']       # Person names
                self.encodings = data['arr_2']   # Encoded labels (numbers)
                self.face_paths = data['arr_3']  # Paths to face images
                
                # Normalize input vectors if needed
                self.embeddings = self.in_encoder.transform(self.embeddings)
                
                # Fit the label encoder with the names
                self.out_encoder.fit(self.names)
                
                # Load the pre-trained model
                self.model = joblib.load('svm_face_model.pkl')
                print("Loaded existing model and embeddings successfully")
                return True
            else:
                raise ValueError("Embeddings file does not have the expected structure")
        except Exception as e:
            print(f"Warning: Could not load model or embeddings: {e}")
            print("You'll need to train the model after adding faces")
            return False
    
    def get_database_info(self):
        """Get information about the current database"""
        if self.embeddings is None or self.names is None:
            return {'status': 'error', 'message': 'No database loaded'}
        
        # Get unique names and count occurrences
        unique_names = np.unique(self.names)
        name_counts = {name: int(np.sum(self.names == name)) for name in unique_names}
        
        info = {
            'status': 'ok',
            'total_faces': len(self.names),
            'total_people': len(unique_names),
            'people': name_counts,
            'embedding_size': self.embeddings.shape[1],
            'model_type': type(self.model).__name__ if self.model else "None"
        }
        
        return info
    
    # --- Face Processing Functions ---
    
    def extract_face(self, img, face_box, required_size=(160, 160)):
        """Extract face from an image"""
        x1, y1, width, height = face_box
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height

        face = img[y1:y2, x1:x2]
        if face.size == 0:  # Check if face crop is empty
            return None
        
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = np.asarray(image)
        return face_array
    
    def get_embeddings_batch(self, face_pixels_list, batch_size=32):
        """Get face embeddings using FaceNet in batches for GPU efficiency"""
        embeddings = []
        
        # Process in batches for better GPU utilization
        for i in range(0, len(face_pixels_list), batch_size):
            batch = face_pixels_list[i:i+batch_size]
            batch = np.array(batch).astype('float32')

            # Use GPU-accelerated Keras
            if self.using_gpu:
                with tf.device('/GPU:0'):  # Use the first GPU device
                    batch_embeddings = self.facenet_model.embeddings(batch)
            else:
                batch_embeddings = self.facenet_model.embeddings(batch)

            embeddings.extend(batch_embeddings)

        return np.array(embeddings)
    
    def get_embedding(self, face_pixels):
        """Get face embeddings using FaceNet for a single face"""
        face_pixels = face_pixels.astype('float32')
        samples = np.expand_dims(face_pixels, axis=0)
        
        # Add embedding verification
        embedding = self.facenet_model.embeddings(samples)[0]
        if not self.embedding_verified:
            print(f"DEBUG: Embedding shape: {embedding.shape}")  # Should be (512,)
            self.embedding_verified = True
        return embedding
    
    def findDistance(self, p1, p2):
        """Calculate distance between two points"""
        x1, y1 = p1
        x2, y2 = p2
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    def get_horizontal_position(self, bbox, frame_width):
        """Return 'left', 'right', or 'front' based on face bbox center."""
        x, y, w, h = bbox
        center_x = x + w / 2
        norm_pos = (center_x - frame_width / 2) / (frame_width / 2)
        if abs(norm_pos) <= 0.2:
            return "front"
        elif norm_pos < 0:
            return "left"  
        else:
            return "right"
    
    # --- Face Recognition Functions ---
    
    def recognize_faces(self, frame):
        """Recognize faces in a frame"""
        if self.processing_frame:
            return None
            
        self.processing_frame = True
        
        try:
            # Convert frame to RGB for detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.detector.detect_faces(rgb_frame)
            
            recognition_results = []
            extracted_faces = []
            bboxes = []
            depths = []
            
            # Process detected faces
            for face_dict in results:
                bbox = face_dict['box']
                
                # Calculate depth
                face_landmarks = face_dict['keypoints']
                point_left = (face_landmarks['left_eye'][0], face_landmarks['left_eye'][1])
                point_right = (face_landmarks['right_eye'][0], face_landmarks['right_eye'][1])
                
                w_pixels = self.findDistance(point_left, point_right)
                
                # Depth estimation
                W = 11.4  # Known distance in cm between eyes
                f = 700  # Approximate focal length
                depth = (W * f) / w_pixels
                
                bboxes.append(bbox)
                depths.append(depth)
                
                # Only recognize faces within distance threshold
                if depth < 100:
                    face_array = self.extract_face(rgb_frame, bbox)
                    if face_array is not None:
                        extracted_faces.append(face_array)
                    else:
                        extracted_faces.append(None)
                else:
                    extracted_faces.append(None)
            
            # Process all valid faces at once for GPU efficiency
            valid_faces = [face for face in extracted_faces if face is not None]
            valid_indices = [i for i, face in enumerate(extracted_faces) if face is not None]
            
            if valid_faces and self.model is not None:
                # Get embeddings for all valid faces in one batch
                face_embeddings = self.get_embeddings_batch(valid_faces)
                face_embeddings = self.in_encoder.transform(face_embeddings)
                
                # Predict identities
                yhat_classes = self.model.predict(face_embeddings)
                yhat_probs = self.model.predict_proba(face_embeddings)
                
                # Associate predictions with their original faces
                valid_predictions = []
                for i, (class_index, probs) in enumerate(zip(yhat_classes, yhat_probs)):
                    class_probability = probs[class_index] * 100
                    predict_name = self.out_encoder.inverse_transform([class_index])[0]
                    
                    # Apply probability threshold
                    if class_probability < self.PROBABILITY_THRESHOLD:
                        predict_name = "Unknown"
                    
                    valid_predictions.append({
                        'name': predict_name,
                        'probability': class_probability
                    })
            
                # Create final results including all faces
                for i in range(len(extracted_faces)):
                    bbox = bboxes[i]
                    depth = depths[i]
                    
                    if i in valid_indices:
                        # This was a valid face that was processed
                        valid_idx = valid_indices.index(i)
                        recognition_results.append({
                            'name': valid_predictions[valid_idx]['name'],
                            'probability': valid_predictions[valid_idx]['probability'],
                            'bbox': bbox,
                            'depth': depth
                        })
                    
            return recognition_results
        
        except Exception as e:
            print(f"Error in face recognition: {e}")
            return None
        
        finally:
            self.processing_frame = False
    
    # --- Training Functions ---
    
    def load_dataset(self, directory):
        """Load dataset from directory"""
        faces = []
        labels = []
        paths = []
        
        # Enumerate the subfolders (person names)
        for person_name in os.listdir(directory):
            person_dir = os.path.join(directory, person_name)
            
            # Skip if not a directory
            if not os.path.isdir(person_dir):
                continue
                
            # Enumerate images for this person
            for filename in os.listdir(person_dir):
                # Skip non-image files
                if not filename.endswith(('.jpg', '.jpeg', '.png')):
                    continue
                    
                # Load image
                img_path = os.path.join(person_dir, filename)
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Warning: Could not load {img_path}")
                    continue
                    
                # Convert to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Detect faces
                results = self.detector.detect_faces(image_rgb)
                if results:
                    # Get the largest face
                    face = max(results, key=lambda x: x['box'][2] * x['box'][3])
                    bbox = face['box']
                    
                    # Extract face
                    face_array = self.extract_face(image_rgb, bbox)
                    if face_array is not None:
                        faces.append(face_array)
                        labels.append(person_name)
                        paths.append(img_path)
        
        return np.array(faces), np.array(labels), np.array(paths)
    
    def train_model(self):
        """Train a new model using the dataset"""
        print("\n--- Training New Model ---")
        
        # Check if dataset exists
        dataset_dir = os.path.join(self.dataset_folder, 'train')
        if not os.path.exists(dataset_dir):
            print(f"Error: Dataset directory '{dataset_dir}' does not exist.")
            return False
        
        # Check if there are any people in the dataset
        if len(os.listdir(dataset_dir)) == 0:
            print("Error: No people in the dataset.")
            return False
        
        print("Loading dataset...")
        faces, labels, paths = self.load_dataset(dataset_dir)
        
        if len(faces) == 0:
            print("Error: No valid face images found in the dataset.")
            return False
        
        print(f"Loaded {len(faces)} images for {len(np.unique(labels))} people.")
        
        # Get face embeddings with batch processing for better GPU utilization
        print("Generating face embeddings in batches for GPU optimization...")
        batch_size = 32  # Adjust based on your GPU memory
        new_embeddings = self.get_embeddings_batch(faces, batch_size)
        
        # Store all the data
        self.embeddings = np.array(new_embeddings)
        self.names = labels
        
        # Normalize embeddings
        print("Normalizing embeddings...")
        self.embeddings = self.in_encoder.transform(self.embeddings)
        
        # Encode labels
        print("Encoding labels...")
        self.out_encoder.fit(self.names)
        self.encodings = self.out_encoder.transform(self.names)
        
        # Store face image paths
        self.face_paths = paths
        
        # Train model
        print("Training SVM model...")
        self.model = SVC(kernel='linear', probability=True)
        self.model.fit(self.embeddings, self.encodings)
        
        # Save the model and embeddings
        print("Saving model and embeddings...")
        joblib.dump(self.model, 'svm_face_model.pkl')
        
        # Save in the new format with four arrays
        np.savez_compressed('embeddings.npz', 
                           self.embeddings,        # arr_0: Face embeddings 
                           self.names,             # arr_1: Person names
                           self.encodings,         # arr_2: Encoded label numbers
                           self.face_paths)        # arr_3: Paths to face images
        # save the label encoder
        joblib.dump(self.out_encoder, 'label_encoder.pkl')

        #save the normalizer
        joblib.dump(self.in_encoder, 'normalizer.pkl')

        #save dataset.npz
        np.savez_compressed('dataset.npz',
                           faces=faces,            # arr_0: Face images
                           labels=labels,          # arr_1: Person names
                           paths=paths)            # arr_2: Paths to face images
        
        # --- CHANGED: Evaluate on test set instead of validation set ---
        test_dir = os.path.join(self.dataset_folder, 'test')
        if os.path.exists(test_dir) and len(os.listdir(test_dir)) > 0:
            print("Evaluating on test set...")
            test_faces, test_labels, _ = self.load_dataset(test_dir)
            
            if len(test_faces) > 0:
                # Get embeddings for test set using batch processing
                test_embeddings = self.get_embeddings_batch(test_faces, batch_size)
                test_embeddings = self.in_encoder.transform(test_embeddings)
                
                # Encode test labels
                test_encodings = self.out_encoder.transform(test_labels)
                
                # Evaluate
                test_pred = self.model.predict(test_embeddings)
                test_accuracy = accuracy_score(test_encodings, test_pred)
                print(f"Test accuracy: {test_accuracy:.2%}")
                #  add training accuracy
                train_pred = self.model.predict(self.embeddings)
                train_accuracy = accuracy_score(self.encodings, train_pred)
                print(f"Training accuracy: {train_accuracy:.2%}")
                

                # add calcification report 
                print("Classification Report:")
                from sklearn.metrics import classification_report
                print(classification_report(test_encodings, test_pred))


                

                # add convision matrix
                cm = confusion_matrix(test_encodings, test_pred)
                print("Confusion Matrix:")
                print(cm)
        
        print("Model training completed successfully!")
        self.database_modified = False
        return True
    
    def add_face(self, person_name, frame):
        """Add a new face to the database"""
        try:
            # Convert frame to RGB for processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            results = self.detector.detect_faces(rgb_frame)
            
            if not results:
                print("No face detected in the image")
                return False
            
            # Get the largest face
            face = max(results, key=lambda x: x['box'][2] * x['box'][3])
            bbox = face['box']
            
            # Extract face with some margin
            margin = 20
            x, y, w, h = bbox
            x_extended = max(0, x - margin)
            y_extended = max(0, y - margin)
            w_extended = min(w + 2*margin, frame.shape[1] - x_extended)
            h_extended = min(h + 2*margin, frame.shape[0] - y_extended)
            
            face_crop = frame[y_extended:y_extended+h_extended, x_extended:x_extended+w_extended]
            
            if face_crop.shape[0] <= 0 or face_crop.shape[1] <= 0:
                print("Invalid face crop dimensions")
                return False
            
            # Create directories if needed
            train_dir = os.path.join(self.dataset_folder, 'train', person_name)
            test_dir = os.path.join(self.dataset_folder, 'val', person_name)
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)
            
            # Count existing files to determine next number
            existing_files = len([f for f in os.listdir(train_dir) if f.endswith('.jpg')])
            
            # Save to training directory (80%)
            filename = f"{person_name}_{existing_files:03d}.jpg"
            file_path = os.path.join(train_dir, filename)
            cv2.imwrite(file_path, face_crop)
            
            # Increment for test file
            existing_files += 1
            
            # Save to test directory (20%)
            if existing_files % 5 == 0:  # Every 5th image goes to test
                filename = f"{person_name}_{existing_files//5:03d}.jpg"
                file_path = os.path.join(test_dir, filename)
                cv2.imwrite(file_path, face_crop)
            
            self.database_modified = True
            print(f"Face saved for {person_name}")
            return True
            
        except Exception as e:
            print(f"Error adding face: {e}")
            return False
        
    def record_new_person(self):
        """Record video for a new person with name verification and image capture"""
        # Use existing camera instead of reinitializing
        if self.camera is None or not self.camera.isOpened():
            print("Camera not available for recording")
            self.speak("Camera not available for recording")
            return False
        
        # Name capture and verification phase
        name_confirmed = False
        person_name = ""
        attempts = 0
        MAX_ATTEMPTS = 3

        # Use speech recognition if available, otherwise use text input
        while not name_confirmed and attempts < MAX_ATTEMPTS and not self.exit_flag:
            if self.speech_to_text is not None:
                # Get name input via speech recognition
                self.speak("Hold 1 and say the person's name")
                try:
                    audio_bytes = self.speech_to_text.record_until_release(key='1')
                    face_to_add = self.speech_to_text.transcribe(audio_bytes)
                    person_name = face_to_add.strip()
                except Exception as e:
                    print(f"Error in speech recognition: {e}")
                    self.speak("Speech recognition error. Please try again.")
                    attempts += 1
                    continue

                if not person_name:
                    self.speak("Name not recognized. Please try again.")
                    attempts += 1
                    continue
            else:
                # Fallback to text input when speech recognition is not available
                print("Speech recognition not available. Using text input.")
                self.speak("Speech recognition not available. Please type the person's name.")
                person_name = input("Enter person's name: ").strip()
                
                if not person_name:
                    self.speak("No name provided. Registration canceled.")
                    return False
            
            name_confirmed = True

            # Name confirmation phase
            self.speak(f"You said: {person_name}. Press 1 to confirm or 0 to cancel. Timeout in 5 seconds.")
            confirmation_start = time.time()
            TIMEOUT = 5

            while (time.time() - confirmation_start) < TIMEOUT and not self.exit_flag:
                ret, frame = self.camera.read()
                if not ret:
                    print("Camera read error during confirmation")
                    continue

                # Prepare confirmation display
                display_frame = cv2.flip(frame, 1)
                timer = TIMEOUT - int(time.time() - confirmation_start)
                
                # UI elements
                cv2.putText(display_frame, "CONFIRM NAME", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, person_name, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, f"1: Confirm | 0: Cancel | Time: {timer}s",
                            (10, display_frame.shape[0]-40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow('Face Recognition System', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('1'):
                    name_confirmed = True
                    self.speak(f"Name confirmed: {person_name}")
                    break
                elif key == ord('0'):
                    self.speak("Registration canceled")
                    return False

            if not name_confirmed:
                attempts += 1
                if attempts < MAX_ATTEMPTS:
                    self.speak("Confirmation timed out. Please try again.")
                else:
                    self.speak("Too many failed attempts. Canceling registration.")
                    return False

        if not name_confirmed:
            return False

        # Create dataset directories
        train_dir = os.path.join('dataset', 'train', person_name)
        test_dir = os.path.join('dataset', 'test', person_name)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # Image capture phase
        frames = []
        frame_count = 0
        total_frames_needed = 40
        recording = False
        capture_delay = 0  # Add delay between captures to avoid duplicate frames
        
        self.speak(f"Press 1 to start capturing 40 images for {person_name}. Press 0 to cancel.")

        while not self.exit_flag and frame_count < total_frames_needed:
            ret, frame = self.camera.read()
            if not ret:
                print("Camera read error during capture")
                continue

            display_frame = cv2.flip(frame, 1)

            if recording and capture_delay <= 0:
                # Face detection and processing
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                try:
                    results = self.detector.detect_faces(rgb_frame)
                except Exception as e:
                    print(f"Face detection error: {e}")
                    results = []

                if results:
                    face = max(results, key=lambda x: x['box'][2] * x['box'][3])
                    x, y, w, h = face['box']
                    x, y = max(0, x), max(0, y)

                    # Draw mirrored bounding box for display
                    flipped_x = display_frame.shape[1] - x - w
                    cv2.rectangle(display_frame, (flipped_x, y),
                                (flipped_x + w, y + h), (0, 255, 0), 2)

                    # Depth calculation for quality control
                    try:
                        face_landmarks = face['keypoints']
                        point_left = face_landmarks['left_eye']
                        point_right = face_landmarks['right_eye']
                        w_pixels = self.findDistance(point_left, point_right)
                        depth = (11.4 * 700) / w_pixels if w_pixels > 0 else 999
                    except:
                        depth = 999  # Default to far if calculation fails

                    if depth < 100:  # Good distance
                        # Capture face region with margin
                        margin = 20
                        x_ext = max(0, x - margin)
                        y_ext = max(0, y - margin)
                        w_ext = min(w + 2*margin, frame.shape[1] - x_ext)
                        h_ext = min(h + 2*margin, frame.shape[0] - y_ext)
                        
                        face_crop = frame[y_ext:y_ext+h_ext, x_ext:x_ext+w_ext]
                        
                        if face_crop.size > 0:
                            frames.append(face_crop)
                            frame_count += 1
                            capture_delay = 3  # Wait 3 frames before next capture

                            # Progress display
                            progress_text = f"Captured: {frame_count}/{total_frames_needed}"
                            cv2.putText(display_frame, progress_text, (10, 100),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            

                            # Voice feedback every 10 frames
                            if frame_count % 10 == 0:
                                self.speak(f"{frame_count} images captured")

                            if frame_count >= total_frames_needed:
                                self.speak("Image capture complete")
                                break
                    else:
                        depth_warning = f"Too far! Move closer ({int(depth)}cm)"
                        cv2.putText(display_frame, depth_warning, (10, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    cv2.putText(display_frame, "No face detected - look at camera", (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Decrease capture delay
            if capture_delay > 0:
                capture_delay -= 1

            # UI elements
            mode_text = "RECORDING" if recording else "READY"
            status_color = (0, 255, 0) if recording else (255, 255, 0)
            cv2.putText(display_frame, f"MODE: {mode_text}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(display_frame, f"Subject: {person_name}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if recording:
                cv2.putText(display_frame, f"Progress: {frame_count}/{total_frames_needed}", (10, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            help_text = "1: Stop Recording | 0: Cancel" if recording else "1: Start Recording | 0: Cancel"
            cv2.putText(display_frame, help_text, (10, display_frame.shape[0]-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Face Recognition System', display_frame)

            # Handle key inputs
            key = cv2.waitKey(1) & 0xFF
            if key == ord('0'):  # Changed from 'q'
                self.speak("Registration canceled")
                return False
            elif key == ord('1'):  # Changed from 'r'
                recording = not recording
                if recording:
                    self.speak("Recording started. Look at the camera.")
                else:
                    self.speak("Recording paused")

        # Save captured images if we have enough
        if frame_count >= total_frames_needed:
            try:
                # Split into train/test sets (80/20)
                train_count = int(len(frames) * 0.8)
                
                # Save training images
                for i, img in enumerate(frames[:train_count]):
                    filename = f"{person_name}_{i:03d}.jpg"
                    filepath = os.path.join(train_dir, filename)
                    cv2.imwrite(filepath, img)
                
                # Save test images
                for i, img in enumerate(frames[train_count:]):
                    filename = f"{person_name}_{i:03d}.jpg"
                    filepath = os.path.join(test_dir, filename)
                    cv2.imwrite(filepath, img)
                
                self.speak(f"Successfully saved {len(frames)} images for {person_name}. You can now train the model.")
                self.database_modified = True
                return True
                
            except Exception as e:
                print(f"Error saving images: {e}")
                self.speak("Error occurred while saving images")
                return False
        else:
            self.speak("Not enough images captured. Registration incomplete.")
            return False
    
    # --- Camera Functions ---
    
    def start_camera(self):
        """Start the camera for face recognition"""
        try:
            self.camera = cv2.VideoCapture(self.camera_index,cv2.CAP_DSHOW)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.camera.set(cv2.CAP_PROP_FPS, 30)

            
            if not self.camera.isOpened():
                print(f"Face Recognition: Failed to open camera!")
                return None
            return True
            
        except Exception as e:
            print(f"Error starting camera: {e}")
            self.speak("Error occurred while starting camera")
            return False
    
    def stop_camera(self):
        """Stop the camera"""
        if self.camera:
            self.camera.release()
            self.camera = None
            print("Camera stopped")
    
    # --- Main Application ---
    
    def run_Face_Recognition(self):
        """Run the unified face recognition system, with mode switching support"""
        # Check if already processing
        if self.processing:
            print("Face recognition is already running")
            return None
            
        self.processing = True  # Set flag when starting
        
        # --- EDITED: Add welcome message with menu ---
        self.speak("Face Recognition system activated.")
        menu_text = "Controls: Press 1 for Add Face, Press 2 for Train, Press 3 for Info, Press 4 for Clear, Press Enter to Analyze, Press 0 for Quit"
        print(menu_text)
        self.speak(menu_text)
        
        self.exit_flag = False  # Reset the exit flag
        switch_mode = None  # Initialize switch mode variable
        
        try:
            if not self.start_camera():
                print("Failed to start camera")
                self.processing = False
                return None

            face_to_add = None
            last_frame = None

            print("Camera started successfully")
            camera_started_announced = False  # Add this flag

            while not self.exit_flag:
                ret, frame = self.camera.read()
                if not ret:
                    print("Error reading from camera")
                    self.speak("Error occurred while reading from camera")
                    break

                display_frame = cv2.flip(frame, 1)
                last_frame = frame.copy()

                # If we're adding a face, show prompt
                if face_to_add:
                    cv2.putText(display_frame, f"Adding face for {face_to_add}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display_frame, "Press 's' to save, '4' to cancel", (20, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    # Show mode message on camera feed
                    cv2.putText(display_frame, "FACE RECOGNITION SYSTEM", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display mode-switch instructions
                instructions = "Enter: Analyze | 1: Add Face | 2: Train | 3: Info | 4: Clear | 0: Quit"
                cv2.putText(display_frame, instructions, (10, display_frame.shape[0] - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                cv2.imshow('Face Recognition System', display_frame)

                # add delay to allow camera to stabilize
                time.sleep(0.1)  # Adjust as needed for camera stabilization

                # --- Announce camera started after first frame is shown ---
                if not camera_started_announced:
                    self.speak("camera started.")
                    camera_started_announced = True

                key = cv2.waitKey(1) & 0xFF

                if key == ord('0'):  # Changed from 'q'
                    self.speak("Shutting down system")
                    break

                elif key == 13:  # Enter key
                    print("\n--- Analyzing current frame ---")
                    self.speak("Analyzing current frame")
                    
                    if last_frame is not None:
                        # Recognize faces in the current frame
                        results = self.recognize_faces(last_frame)
                        if results:
                            self.current_results = results
                            self.display_results = True
                            valid_faces = [r for r in results if r['name'].lower() != "unknown"]
                            for i, face in enumerate(valid_faces):
                                print(f"Face {i+1}: {face['name']} ({face['probability']:.1f}% confidence)")
                            self.speak_results(results)
                            # --- Add this line ---
                            self.speak_face_locations(results, last_frame.shape[1])
                        else:
                            print("No results from recognition or error occurred")

                elif key == ord('1') and not face_to_add:  # Changed from 'a'
                    self.record_new_person()
                
                elif key == ord('4'):  # Changed from 'c'
                    if face_to_add:
                        print("Cancelled face addition")
                        self.speak("Cancelled face addition")
                        face_to_add = None
                    else:
                        print("Cleared results")
                        self.speak("Cleared results")
                        self.current_results = []
                        self.display_results = False
                        
                elif key == ord('2'):  # Changed from 't'
                    print("Training model...")
                    self.speak("Training model, please wait")
                    success = self.train_model()
                    if success:
                        self.speak("Model training completed successfully")
                    else:
                        self.speak("Model training failed")
                        
                elif key == ord('3'):  # Changed from 'i'
                    self.speak("Getting database information")
                    info = self.get_database_info()
                    if info:
                        print("\n=== Database Information ===")
                        print(f"Total Faces: {info.get('total_faces', 'N/A')}")
                        print(f"Total People: {info.get('total_people', 'N/A')}")
                        print("People in database:")
                        for person, count in info.get('people', {}).items():
                            print(f"  - {person}: {count} images")
                        print("=========================\n")
                        self.speak_database_info(info)

        except Exception as e:
            print(f"Error in face recognition: {e}")
            import traceback
            traceback.print_exc()
            return None
            
        finally:
            # --- EDITED: Add goodbye message ---
            self.speak("Exiting Face Recognition system.")
            time.sleep(1)  # Give message time to play
            
            print("Cleaning up face recognition resources...")
            self.stop_camera()
            cv2.destroyAllWindows()
            self.processing = False  # Reset flag when done
            
            # Only set exit_flag to True when we're truly exiting
            if switch_mode is None:
                self.exit_flag = True
                
            return switch_mode  # Return which mode to switch to (if any)

    def test_on_new_data(self, test_dir=None):
        """
        Test the trained model on a new test dataset and print accuracy.
        test_dir: path to the test dataset (default: dataset/test)
        """
        if test_dir is None:
            test_dir = os.path.join(self.dataset_folder, 'test')
        if not os.path.exists(test_dir):
            print(f"Test directory '{test_dir}' does not exist.")
            return

        print(f"Loading test dataset from {test_dir} ...")
        faces, labels, _ = self.load_dataset(test_dir)
        if len(faces) == 0:
            print("No valid face images found in the test dataset.")
            return

        print(f"Loaded {len(faces)} test images for {len(np.unique(labels))} people.")

        # Get embeddings for test set
        batch_size = 32
        test_embeddings = self.get_embeddings_batch(faces, batch_size)
        test_embeddings = self.in_encoder.transform(test_embeddings)

        # Encode test labels
        test_encodings = self.out_encoder.transform(labels)

        # Predict
        test_pred = self.model.predict(test_embeddings)
        test_accuracy = accuracy_score(test_encodings, test_pred)
        print(f"Test accuracy: {test_accuracy:.2%}")

if __name__ == "__main__":
    # Create and run the unified face recognition system
    face_system = UnifiedFaceRecognition()
    face_system.run_Face_Recognition()