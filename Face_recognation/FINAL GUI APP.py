import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import numpy as np
from mtcnn.mtcnn import MTCNN
from PIL import Image, ImageTk
from numpy import savez_compressed, asarray, expand_dims, load
import joblib
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from keras_facenet import FaceNet
import threading
import queue
from os import listdir
from os.path import isdir
from PIL import Image
from numpy import savez_compressed, asarray, expand_dims
import shutil
import random
from sklearn.model_selection import train_test_split


class ModernFaceGUI:
    def __init__(self, master):
        self.master = master
        master.title("FaceTrain Studio")
        master.geometry("900x800")
        master.configure(bg='#F5F7FB')
        self.train_accuracy = 0.0
        self.test_accuracy = 0.0
        self.classification_report = "No classification report available."

        self.confusion_matrix_image = None  # To hold the current confusion matrix image


        # Initialize paths FIRST
        self.initialize_paths()  # This must come before load_existing_models()
        
         # Configure models and load existing data
        self.detector = MTCNN()
        self.facenet = FaceNet()
        self.in_encoder = Normalizer(norm='l2')
        
        # Then load existing models
        self.load_existing_models()  # Now model_path exists

        # Modern color scheme
        self.colors = {
            'background': '#F5F7FB',
            'primary': '#4A90E2',
            'secondary': '#7ED321',
            'accent': '#FF6B6B',
            'text': '#2D3436',
            'card': '#FFFFFF'
        }
        
        # Threading and queue setup
        self.progress_queue = queue.Queue()
        self.training_thread = None
        
        # Initialize models and paths
        self.detector = MTCNN()
        self.facenet = FaceNet()
        self.in_encoder = Normalizer(norm='l2')
        self.initialize_paths()
        
        # Configure styles and create UI
        self.configure_styles()
        self.create_widgets()
        
        # Load models if they exist
        self.load_existing_models()
        
        # Store the current person name
        self.current_person = None

    def configure_styles(self):
        self.style = ttk.Style()
        self.style.theme_create('modern', settings={
            'TLabel': {
                'configure': {
                    'foreground': self.colors['text'],
                    'background': self.colors['background'],
                    'font': ('Segoe UI', 10)
                }
            },
            'TButton': {
                'configure': {
                    'foreground': 'white',
                    'background': self.colors['primary'],
                    'borderwidth': 0,
                    'font': ('Segoe UI', 10, 'bold'),
                    'padding': (15, 8),
                    'relief': 'flat'
                },
                'map': {
                    'background': [('active', self.colors['secondary'])]
                }
            },
            'TEntry': {
                'configure': {
                    'fieldbackground': self.colors['card'],
                    'foreground': self.colors['text'],
                    'insertcolor': self.colors['text'],
                    'padding': 8,
                    'relief': 'flat'
                }
            },
            'TNotebook': {
                'configure': {
                    'tabmargins': [2, 5, 2, 0],
                    'background': self.colors['background']
                }
            },
            'TNotebook.Tab': {
                'configure': {
                    'padding': [10, 5],
                    'background': self.colors['card'],
                    'foreground': self.colors['text']
                },
                'map': {
                    'background': [('selected', self.colors['primary'])],
                    'foreground': [('selected', 'white')],
                    'expand': [('selected', [1, 1, 1, 0])]
                }
            }
        })
        self.style.theme_use('modern')

    def initialize_paths(self):
        """Properly initialize all path attributes"""
        os.makedirs('dataset/train', exist_ok=True)
        os.makedirs('dataset/val', exist_ok=True)
        
        # Define all path attributes
        self.dataset_path = 'dataset.npz'
        self.embeddings_path = 'embeddings.npz'
        self.model_path = 'svm_face_model.pkl'  # This creates the attribute
        self.encoder_path = 'label_encoder.pkl'
        self.normalizer_path = 'normalizer.pkl'

    def create_widgets(self):
        self.main_frame = ttk.Frame(self.master)
        self.main_frame.pack(fill='both', expand=True, padx=40, pady=40)
        
        self.create_header()
        
        # Create a notebook for tabs
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill='both', expand=True, pady=10)
        
        # Add Person Tab
        self.add_person_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.add_person_tab, text="Add Person")
        
        # Test Recognition Tab
        self.test_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.test_tab, text="Test Recognition")
        
        # Add content to tabs
        self.create_add_person_tab()
        self.create_test_tab()

        # Performance Tab
        self.performance_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.performance_tab, text="Performance")
        self.create_performance_tab()
        
        # Progress bar and status at the bottom
        self.create_progress_section()

    def create_header(self):
        header_frame = ttk.Frame(self.main_frame)
        header_frame.pack(fill='x', pady=(0, 20))
        
        self.logo_label = ttk.Label(header_frame, 
                                  text="👤 FaceTrain Studio", 
                                  font=('Segoe UI', 24, 'bold'),
                                  foreground=self.colors['primary'])
        self.logo_label.pack(side='left')
        
        ttk.Label(header_frame, 
                 text="v2.1 • AI Training Platform",
                 foreground=self.colors['text']).pack(side='left', padx=10)
        
        # Show stats
        self.stats_frame = ttk.Frame(header_frame)
        self.stats_frame.pack(side='right')
        
        self.stat_items = {
            'dataset': ('📁 Dataset', '0 images'),
            'people': ('👥 People', '0'),
            'accuracy': ('🎯 Accuracy', 'N/A')
        }
        
        for key, (title, value) in self.stat_items.items():
            label_frame = ttk.Frame(self.stats_frame)
            label_frame.pack(side='left', padx=10)
            ttk.Label(label_frame, text=title, font=('Segoe UI', 8)).pack(anchor='w')
            ttk.Label(label_frame, text=value, font=('Segoe UI', 10, 'bold')).pack(anchor='w')

    def create_add_person_tab(self):
        content_frame = ttk.Frame(self.add_person_tab, style='Card.TFrame')
        content_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left side - Input fields
        left_frame = ttk.Frame(content_frame)
        left_frame.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        
        ttk.Label(left_frame, 
                 text="Add New Identity", 
                 font=('Segoe UI', 14, 'bold')).pack(anchor='w', pady=(0, 15))
        
        # Name input
        name_frame = ttk.Frame(left_frame)
        name_frame.pack(fill='x', pady=5)
        ttk.Label(name_frame, text="Full Name:").pack(side='left')
        self.name_entry = ttk.Entry(name_frame, width=30)
        self.name_entry.pack(side='left', padx=10, fill='x', expand=True)
        
        # Folder selection
        self.folder_frame = ttk.Frame(left_frame)
        self.folder_frame.pack(fill='x', pady=5)
        
        self.folder_btn = ttk.Button(self.folder_frame, 
                                    text="📂 Select Image Folder", 
                                    command=self.select_folder)
        self.folder_btn.pack(fill='x')
        
        self.folder_label = ttk.Label(self.folder_frame, 
                                     text="No folder selected",
                                     foreground='#666666')
        self.folder_label.pack(anchor='w', pady=5)
        
        # Process button
        self.process_btn = ttk.Button(left_frame, 
                                     text="🎓 Train AI Model", 
                                     style='Accent.TButton',
                                     command=self.start_training_process)
        self.process_btn.pack(fill='x', pady=10)
        
        # Training information
        self.info_text = tk.Text(left_frame, height=8, width=40, 
                               bg=self.colors['card'], fg=self.colors['text'],
                               font=('Segoe UI', 9))
        self.info_text.pack(fill='both', expand=True, pady=10)
        self.info_text.insert('1.0', "Instructions:\n\n"
        "1. Enter the person's name\n"
        "2. Select image folder\n"
        "3. Click 'Train AI Model'\n\n"
        "The system will add new person in system.")
        self.info_text.config(state='disabled')
        
        # Right side - Preview
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side='right', fill='both', expand=True, padx=10, pady=10)
        
        ttk.Label(right_frame, 
                 text="Image Preview", 
                 font=('Segoe UI', 14, 'bold')).pack(anchor='w', pady=(0, 15))
        
        # Preview canvas
        self.preview_frame = ttk.Frame(right_frame, style='Card.TFrame')
        self.preview_frame.pack(fill='both', expand=True)
        
        self.preview_canvas = tk.Canvas(self.preview_frame, width=300, height=300, 
                                      bg=self.colors['card'], highlightthickness=0)
        self.preview_canvas.pack(pady=10)
        
        self.preview_label = ttk.Label(self.preview_frame, text="No image selected")
        self.preview_label.pack()

    def create_test_tab(self):
        content_frame = ttk.Frame(self.test_tab, style='Card.TFrame')
        content_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left side - Test controls
        left_frame = ttk.Frame(content_frame)
        left_frame.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        
        ttk.Label(left_frame, 
                 text="Test Recognition", 
                 font=('Segoe UI', 14, 'bold')).pack(anchor='w', pady=(0, 15))
        
        # Person selection
        person_frame = ttk.Frame(left_frame)
        person_frame.pack(fill='x', pady=5)
        ttk.Label(person_frame, text="Select Person:").pack(side='left')
        
        self.person_var = tk.StringVar()
        self.person_dropdown = ttk.Combobox(person_frame, textvariable=self.person_var)
        self.person_dropdown.pack(side='left', padx=10, fill='x', expand=True)
        self.update_person_dropdown()
        
        # Test options
        test_options_frame = ttk.Frame(left_frame)
        test_options_frame.pack(fill='x', pady=15)
        
        self.test_image_btn = ttk.Button(test_options_frame, 
                                       text="🖼️ Select Test Image", 
                                       command=self.select_test_image)
        self.test_image_btn.pack(side='left', padx=5, fill='x', expand=True)
        
        # Run test button
        self.run_test_btn = ttk.Button(left_frame, 
                                     text="🔍 Run Recognition Test", 
                                     command=self.run_recognition_test,
                                     state='disabled')
        self.run_test_btn.pack(fill='x', pady=10)
        
        # Right side - Results
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side='right', fill='both', expand=True, padx=10, pady=10)
        
        ttk.Label(right_frame, 
                 text="Recognition Results", 
                 font=('Segoe UI', 14, 'bold')).pack(anchor='w', pady=(0, 15))
        
        # Result display
        self.test_canvas = tk.Canvas(right_frame, width=300, height=300, 
                                   bg=self.colors['card'], highlightthickness=0)
        self.test_canvas.pack(pady=10)
        
        self.results_text = tk.Text(right_frame, height=8, width=40, 
                                  bg=self.colors['card'], fg=self.colors['text'],
                                  font=('Segoe UI', 10))
        self.results_text.pack(fill='both', expand=True, pady=10)
        self.results_text.insert('1.0', "Train a model and run a test to see recognition results.")
        self.results_text.config(state='disabled')

    def create_progress_section(self):
        progress_frame = ttk.Frame(self.main_frame)
        progress_frame.pack(fill='x', pady=10)
        
        self.progress = ttk.Progressbar(progress_frame, 
                                      mode='determinate',
                                      style='Modern.Horizontal.TProgressbar')
        self.progress.pack(fill='x', expand=True)
        
        self.progress_text = ttk.Label(progress_frame, 
                                      text="Ready to train",
                                      foreground=self.colors['text'])
        self.progress_text.pack(pady=5)
        
        self.style.configure('Modern.Horizontal.TProgressbar',
                           thickness=20,
                           troughcolor=self.colors['card'],
                           background=self.colors['primary'])

    def select_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.folder_path = folder_path
            folder_name = os.path.basename(folder_path)
            self.folder_label.config(text=f"Selected: {folder_name}", foreground='#4A90E2')
            
            # Preview first image if available
            for file in os.listdir(folder_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.preview_image(os.path.join(folder_path, file))
                    break
        else:
            self.folder_label.config(text="No folder selected", foreground='#666666')
    def split_and_copy_images(self, source_dir, person_name, split_ratio=0.8):
        """Split images into train/val and copy to dataset directories"""
        # Create output directories
        train_dir = f'dataset/train/{person_name}'
        val_dir = f'dataset/val/{person_name}'
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        
        # Get all image files
        images = [f for f in os.listdir(source_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not images:
            raise ValueError("No images found in selected directory")
        
        # Shuffle and split
        random.shuffle(images)
        split = int(len(images) * split_ratio)
        train_files = images[:split]
        val_files = images[split:]
        
        # Copy files
        for f in train_files:
            src = os.path.join(source_dir, f)
            dst = os.path.join(train_dir, f)
            shutil.copyfile(src, dst)
            
        for f in val_files:
            src = os.path.join(source_dir, f)
            dst = os.path.join(val_dir, f)
            shutil.copyfile(src, dst)

        # # use MTNN to extract faces and save
        for f in train_files:
            src = os.path.join(source_dir, f)
            dst = os.path.join(train_dir, f)
            face = self.extract_face(src)
            if face is not None:
                Image.fromarray(face).save(dst)

        for f in val_files:
            src = os.path.join(source_dir, f)
            dst = os.path.join(val_dir, f)
            face = self.extract_face(src)
            if face is not None:
                Image.fromarray(face).save(dst)
            
        return len(train_files), len(val_files)        

    def start_training_process(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showerror("Error", "Please enter a name first!")
            return
        
        if not hasattr(self, 'folder_path') or not self.folder_path:
            messagebox.showerror("Error", "Please select a folder first!")
            return

        self.current_person = name

        try:
            self.process_btn.config(state='disabled')
            self.progress_queue.queue.clear()
            self.training_thread = threading.Thread(target=self.run_training)
            self.training_thread.start()
            self.master.after(100, self.update_progress_from_queue)
        except Exception as e:
            messagebox.showerror("Error", f"🚨 Error: {str(e)}")
            self.update_progress(0, "Error occurred")

    def update_progress_from_queue(self):
        try:
            while True:
                progress_data = self.progress_queue.get_nowait()
                
                if progress_data == 'COMPLETE':
                    self.on_training_complete()
                elif isinstance(progress_data, tuple) and len(progress_data) == 2:
                    self.update_progress(progress_data[0], progress_data[1])
                elif isinstance(progress_data, str) and progress_data.startswith('ERROR:'):
                    messagebox.showerror("Error", progress_data[6:])
                    self.update_progress(0, "Training failed")
                    self.process_btn.config(state='normal')
        except queue.Empty:
            pass
        self.master.after(100, self.update_progress_from_queue)

    def update_progress(self, value, text):
        self.progress['value'] = value
        self.progress_text.config(text=text)
        self.master.update_idletasks()

    def run_training(self):
        try:
            self.progress_queue.put((10, "Initializing training..."))
            
            # Step 1: Split and copy images to train/val
            self.progress_queue.put((20, "Splitting images..."))
            train_count, val_count = self.split_and_copy_images(
                self.folder_path, 
                self.current_person
            )
            
            # Step 2: Process dataset and generate embeddings
            self.progress_queue.put((40, "Processing images..."))
            trainX, trainy, testX, testy = self.process_datasets()
            
            # Step 3: Train model
            self.progress_queue.put((70, "Training model..."))
            train_acc, test_acc = self.train_model(trainX, trainy, testX, testy)
            
            self.last_training_results = {
                'name': self.current_person,
                'train_acc': train_acc,
                'test_acc': test_acc
            }
            
            self.progress_queue.put('COMPLETE')
            
        except Exception as e:
            self.progress_queue.put(f'ERROR: {str(e)}')

    def extract_face(self, filename, required_size=(160, 160)):
        try:
            image = Image.open(filename)
            image = image.convert('RGB')
            pixels = asarray(image)
            results = self.detector.detect_faces(pixels)
            
            if not results:
                print(f"No faces detected in {filename}")
                return None
            
            x1, y1, width, height = results[0]['box']
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(pixels.shape[1], x2)
            y2 = min(pixels.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                print(f"Invalid face box: {filename}")
                return None
            
            face = pixels[y1:y2, x1:x2]
            image = Image.fromarray(face)
            image = image.resize(required_size)
            face_array = asarray(image)
            
            return face_array
            
        except Exception as e:
            print(f"Error extracting face from {filename}: {str(e)}")
            return None
    
    def load_faces(self, directory):
        faces = list()
        for filename in listdir(directory):
            path = directory + filename
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            face = self.extract_face(path)
            if face is not None:
                print(f"Processed: {path}")
                faces.append(face)
        return faces
    
    def load_dataset(self, directory):
        X, y = list(), list()
        for subdir in listdir(directory):
            path = directory + subdir + '/'
            if not isdir(path):
                print(f"Skipping non-directory: {path}")
                continue
            faces = self.load_faces(path)
            labels = [subdir for _ in range(len(faces))]
            print(f">loaded {len(faces)} examples for class: {subdir}")
            X.extend(faces)
            y.extend(labels)
        return asarray(X), asarray(y)
    
    def get_embedding(self, face_pixels):
        try:
            if face_pixels.shape[0] != 160 or face_pixels.shape[1] != 160:
                face_pixels = asarray(Image.fromarray(face_pixels).resize((160, 160)))
            face_pixels = face_pixels.astype('float32')
            samples = expand_dims(face_pixels, axis=0)
            yhat = self.facenet.embeddings(samples)
            return yhat[0]
        except Exception as e:
            print(f"Error in get_embedding: {str(e)}")
            raise
    def process_datasets(self):
        """Process all images in dataset/train and dataset/val with improved error handling"""
        try:
            # Load datasets
            trainX, trainy = self.load_dataset('dataset/train/')
            testX, testy = self.load_dataset('dataset/val/')
            
            # Ensure arrays are properly shaped
            trainX = asarray(trainX)
            testX = asarray(testX)
            
            # Ensure labels are 1D numpy arrays without path prefixes
            trainy = asarray([label.split('/')[-1] if isinstance(label, str) and '/' in label else label for label in trainy]).ravel()
            testy = asarray([label.split('/')[-1] if isinstance(label, str) and '/' in label else label for label in testy]).ravel()
            
            print(f"Dataset loaded: trainX shape={trainX.shape}, trainy shape={trainy.shape}")
            print(f"Dataset loaded: testX shape={testX.shape}, testy shape={testy.shape}")
            
            # Save raw faces
            savez_compressed('dataset.npz', trainX, trainy, testX, testy)
            
            # Generate embeddings with progress reporting
            self.progress_queue.put((50, "Generating embeddings for training set..."))
            train_embs = []
            for i, face in enumerate(trainX):
                if i % 5 == 0:  # Update progress every 5 images
                    progress = 50 + (i / len(trainX) * 15)  # Progress from 50% to 65%
                    self.progress_queue.put((progress, f"Processing training image {i+1}/{len(trainX)}"))
                emb = self.get_embedding(face)
                train_embs.append(emb)
            
            self.progress_queue.put((65, "Generating embeddings for validation set..."))
            test_embs = []
            for i, face in enumerate(testX):
                if i % 5 == 0:  # Update progress every 5 images
                    progress = 65 + (i / len(testX) * 15)  # Progress from 65% to 80%
                    self.progress_queue.put((progress, f"Processing validation image {i+1}/{len(testX)}"))
                emb = self.get_embedding(face)
                test_embs.append(emb)
            
            # Convert lists to arrays with proper shape
            train_embs = asarray(train_embs)
            test_embs = asarray(test_embs)
            
            print(f"Embeddings generated: train_embs shape={train_embs.shape}")
            print(f"Embeddings generated: test_embs shape={test_embs.shape}")
            
            # Save embeddings
            savez_compressed('embeddings.npz', train_embs, trainy, test_embs, testy)
            return train_embs, trainy, test_embs, testy
            
        except Exception as e:
            print(f"Error in process_datasets: {str(e)}")
            import traceback
            traceback.print_exc()
            self.progress_queue.put(f'ERROR: {str(e)}')
            raise
    def train_model(self, trainX, trainy, testX, testy):
        """Train SVM classifier with in-memory data with improved error handling"""
        try:
            print(f"Training model with shapes: trainX={trainX.shape}, trainy={trainy.shape}, testX={testX.shape}, testy={testy.shape}")
            
            # Normalize
            self.in_encoder = Normalizer(norm='l2')
            trainX_norm = self.in_encoder.transform(trainX)
            testX_norm = self.in_encoder.transform(testX)
            
            # Clean any path prefixes in labels
            cleaned_trainy = np.array([label.split('/')[-1] if isinstance(label, str) and '/' in label else label for label in trainy])
            cleaned_testy = np.array([label.split('/')[-1] if isinstance(label, str) and '/' in label else label for label in testy])
            
            # Encode labels - fit on both train and test to ensure all classes are present
            self.label_encoder = LabelEncoder()
            all_labels = np.concatenate((cleaned_trainy, cleaned_testy))
            self.label_encoder.fit(all_labels)
            trainy_enc = self.label_encoder.transform(cleaned_trainy)
            testy_enc = self.label_encoder.transform(cleaned_testy)
            
            # Print info for debugging
            print(f"Label classes: {self.label_encoder.classes_}")
            print(f"Transformed labels: trainy_enc={trainy_enc.shape}, testy_enc={testy_enc.shape}")
            
            # Train SVM
            self.model = SVC(kernel='linear', probability=True)
            self.model.fit(trainX_norm, trainy_enc)
            
            # Calculate accuracy
            train_preds = self.model.predict(trainX_norm)
            test_preds = self.model.predict(testX_norm)
            
            self.train_accuracy = accuracy_score(trainy_enc, train_preds) * 100
            self.test_accuracy = accuracy_score(testy_enc, test_preds) * 100
            
            # Generate classification report
            self.classification_report = classification_report(
                testy_enc, 
                test_preds,
                target_names=self.label_encoder.classes_,
                zero_division=0  # Handle potential division by zero
            )
            
            # Print report for debugging
            print("Classification Report:")
            print(self.classification_report)
            
            # Save models
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.label_encoder, self.encoder_path)
            joblib.dump(self.in_encoder, self.normalizer_path)
            
            return self.train_accuracy, self.test_accuracy
            
        except Exception as e:
            print(f"Error in train_model: {str(e)}")
            import traceback
            traceback.print_exc()
            return 0.0, 0.0
    
    def create_performance_tab(self):
        content_frame = ttk.Frame(self.performance_tab, style='Card.TFrame')
        content_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Refresh button
        self.refresh_btn = ttk.Button(content_frame, text="🔄 Refresh", command=self.refresh_performance_tab)
        self.refresh_btn.pack(anchor='ne', pady=5)

        # Stats Frame
        stats_frame = ttk.Frame(content_frame)
        stats_frame.pack(fill='x', pady=10)

        # Dataset Stats
        self.dataset_stats_label = ttk.Label(stats_frame, text="Dataset: train=0, test=0")
        self.dataset_stats_label.pack(anchor='w')

        # Accuracy Stats
        self.accuracy_stats_label = ttk.Label(stats_frame, text="Accuracy: train=0.00%, test=0.00%")
        self.accuracy_stats_label.pack(anchor='w')

        # Classification Report
        ttk.Label(content_frame, text="=== Classification Report ===", font=('Segoe UI', 10, 'bold')).pack(anchor='w', pady=10)
        self.report_text = tk.Text(content_frame, wrap=tk.WORD, height=15, bg=self.colors['card'], fg=self.colors['text'])
        self.report_text.pack(fill='both', expand=True)
        self.report_text.insert('1.0', "No classification report available.")
        self.report_text.config(state='disabled')

    def get_dataset_stats(self):
        """Count number of images in train and test datasets"""
        train_count, test_count = 0, 0
        
        # Count training images
        train_dir = 'dataset/train'
        if os.path.isdir(train_dir):
            for person in os.listdir(train_dir):
                person_dir = os.path.join(train_dir, person)
                if os.path.isdir(person_dir):
                    train_count += len([f for f in os.listdir(person_dir) 
                                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        # Count test images
        test_dir = 'dataset/val'
        if os.path.isdir(test_dir):
            for person in os.listdir(test_dir):
                person_dir = os.path.join(test_dir, person)
                if os.path.isdir(person_dir):
                    test_count += len([f for f in os.listdir(person_dir)
                                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        return train_count, test_count    

    def update_performance_data(self):
        """Update all performance metrics in the performance tab"""
        # Dataset stats
        train_count, test_count = self.get_dataset_stats()
        self.dataset_stats_label.config(text=f"Dataset: train={train_count}, test={test_count}")

        # Accuracy stats
        self.accuracy_stats_label.config(text=f"Accuracy: train={self.train_accuracy:.3f}%, test={self.test_accuracy:.3f}%")

        # Classification report
        self.report_text.config(state='normal')
        self.report_text.delete('1.0', tk.END)
        self.report_text.insert('1.0', self.classification_report)
        self.report_text.config(state='disabled')

    def refresh_performance_tab(self):
        try:
            self.load_existing_models()
            if os.path.exists(self.embeddings_path):
                data = load(self.embeddings_path)
                
                # Make sure we extract arrays properly
                trainX = data['arr_0']
                trainy = data['arr_1']
                testX = data['arr_2']
                testy = data['arr_3']
                
                # Ensure features are correctly shaped - reshape if needed
                if len(trainX.shape) == 1:
                    trainX = trainX.reshape(1, -1)
                if len(testX.shape) == 1:
                    testX = testX.reshape(1, -1)
                
                # Ensure labels are 1D arrays (ravel flattens multi-dimensional arrays)
                trainy = trainy.ravel()
                testy = testy.ravel()
                
                # Clean label prefixes if they exist
                cleaned_trainy = np.array([label.split('/')[-1] if isinstance(label, str) and '/' in label else label for label in trainy])
                cleaned_testy = np.array([label.split('/')[-1] if isinstance(label, str) and '/' in label else label for label in testy])
                
                # Normalize inputs
                if hasattr(self, 'in_encoder') and self.in_encoder is not None:
                    trainX_norm = self.in_encoder.transform(trainX)
                    testX_norm = self.in_encoder.transform(testX)
                else:
                    # If no encoder exists, create one
                    self.in_encoder = Normalizer(norm='l2')
                    trainX_norm = self.in_encoder.transform(trainX)
                    testX_norm = self.in_encoder.transform(testX)
                
                # Prepare labels
                if hasattr(self, 'label_encoder') and self.label_encoder is not None:
                    # Fit on both train and test to ensure all classes are represented
                    all_labels = np.concatenate((cleaned_trainy, cleaned_testy))
                    self.label_encoder.fit(all_labels)
                    
                    # Transform labels
                    trainy_enc = self.label_encoder.transform(cleaned_trainy)
                    testy_enc = self.label_encoder.transform(cleaned_testy)
                else:
                    self.label_encoder = LabelEncoder()
                    all_labels = np.concatenate((cleaned_trainy, cleaned_testy))
                    self.label_encoder.fit(all_labels)
                    trainy_enc = self.label_encoder.transform(cleaned_trainy)
                    testy_enc = self.label_encoder.transform(cleaned_testy)
                
                # Calculate performance metrics using model
                if hasattr(self, 'svm_model') and self.svm_model is not None:
                    yhat_train = self.svm_model.predict(trainX_norm)
                    yhat_test = self.svm_model.predict(testX_norm)
                    
                    train_acc = accuracy_score(trainy_enc, yhat_train)
                    test_acc = accuracy_score(testy_enc, yhat_test)
                    
                    report = classification_report(
                        testy_enc, 
                        yhat_test,
                        target_names=self.label_encoder.classes_,
                        zero_division=0  # Handle potential division by zero
                    )
                else:
                    train_acc = 0.0
                    test_acc = 0.0
                    report = "No model available."
            else:
                train_acc = 0.0
                test_acc = 0.0
                report = "No embeddings available."
                
            # Update metrics
            self.train_accuracy = train_acc * 100  # Convert to percentage
            self.test_accuracy = test_acc * 100    # Convert to percentage
            self.classification_report = report
            
            # Update UI
            self.update_performance_data()
            
        except Exception as e:
            self.classification_report = f"Error: {str(e)}"
            self.train_accuracy = 0.0
            self.test_accuracy = 0.0
            self.update_performance_data()
            print(f"Error in refresh_performance_tab: {str(e)}")
            import traceback
            traceback.print_exc()

    def refresh_performance(self):
        """Execute the full performance analysis"""
        try:
            # Load dataset
            data = load(self.embeddings_path)
            
            # Properly assign data
            X = data['arr_0']  # embeddings
            y = data['arr_1']  # labels
            encoded_y = data['arr_2']
            face_paths = data['arr_3']

            # Split data
            trainX, testX, trainy, testy = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Normalize input vectors
            in_encoder = Normalizer(norm='l2')
            trainX_norm = in_encoder.transform(trainX)
            testX_norm = in_encoder.transform(testX)

            # Label encode targets
            out_encoder = LabelEncoder()
            trainy_enc = out_encoder.fit_transform(trainy)
            testy_enc = out_encoder.transform(testy)

            # Train SVM model
            model = SVC(kernel='linear', probability=True)
            model.fit(trainX_norm, trainy_enc)

            # Predictions
            yhat_train = model.predict(trainX_norm)
            yhat_test = model.predict(testX_norm)

            # Calculate metrics
            score_train = accuracy_score(trainy_enc, yhat_train)
            score_test = accuracy_score(testy_enc, yhat_test)

            # Update performance data
            self.train_accuracy = score_train * 100
            self.test_accuracy = score_test * 100
            self.classification_report = classification_report(
                testy_enc, yhat_test, 
                target_names=out_encoder.classes_
            )
            

            # Save models
            joblib.dump(model, self.model_path)
            joblib.dump(out_encoder, self.encoder_path)
            joblib.dump(in_encoder, self.normalizer_path)

            # Update UI
            self.update_performance_data()
            messagebox.showinfo("Refresh Complete", 
                              "Performance metrics updated successfully!")

        except Exception as e:
            messagebox.showerror("Refresh Error", 
                                f"Failed to refresh performance: {str(e)}")   
    def update_performance_data(self):
        """Update all performance metrics in the performance tab"""
        # Get dataset stats
        train_count, test_count = self.get_dataset_stats()
        
        # Update labels
        self.dataset_stats_label.config(
            text=f"Dataset: train={train_count}, test={test_count}"
        )
        self.accuracy_stats_label.config(
            text=f"Accuracy: train={self.train_accuracy:.3f}%, test={self.test_accuracy:.3f}%"
        )
        
        # Update classification report
        self.report_text.config(state='normal')
        self.report_text.delete('1.0', tk.END)
        self.report_text.insert('1.0', self.classification_report)
        self.report_text.config(state='disabled')

               

    def on_training_complete(self):
        self.update_progress(100, "Training complete ✅")
        self.process_btn.config(state='normal')
        self.run_test_btn.config(state='normal')
        self.update_person_dropdown()
        self.update_info_text()
        self.update_stats()
        acc = self.last_training_results['test_acc']
        self.update_performance_data()  # Add this line to refresh performance data
        acc = self.last_training_results['test_acc']
        messagebox.showinfo("Training Complete", 
                          f"Model trained for {self.current_person}!\nAccuracy: {acc:.2f}%")
        

    def update_info_text(self):
        if hasattr(self, 'last_training_results'):
            name = self.last_training_results['name']
            train_acc = self.last_training_results['train_acc']
            test_acc = self.last_training_results['test_acc']
            
            self.info_text.config(state='normal')
            self.info_text.delete('1.0', tk.END)
            self.info_text.insert('1.0', f"Training Results for {name}:\n\n"
                                f"Training Accuracy: {train_acc:.2f}%\n"
                                f"Validation Accuracy: {test_acc:.2f}%\n\n"
                                f"The model has been updated with the new identity.\n"
                                f"You can now test recognition on the Test tab.")
            self.info_text.config(state='disabled')

    def update_stats(self):
        try:
            total_images = 0
            for root, dirs, files in os.walk('dataset'):
                for file in files:
                    if file.endswith(('.jpg', '.jpeg', '.png')):
                        total_images += 1
            
            people = set()
            for dir_name in os.listdir('dataset/train'):
                if os.path.isdir(os.path.join('dataset/train', dir_name)):
                    people.add(dir_name)
            
            accuracy_val = "N/A"
            if hasattr(self, 'last_training_results'):
                accuracy_val = f"{self.last_training_results['test_acc']:.1f}%"
            
            self.stat_items['dataset'] = ('📁 Dataset', f"{total_images} images")
            self.stat_items['people'] = ('👥 People', f"{len(people)}")
            self.stat_items['accuracy'] = ('🎯 Accuracy', accuracy_val)
            
            for widget in self.stats_frame.winfo_children():
                widget.destroy()
            
            for key, (title, value) in self.stat_items.items():
                label_frame = ttk.Frame(self.stats_frame)
                label_frame.pack(side='left', padx=10)
                ttk.Label(label_frame, text=title, font=('Segoe UI', 8)).pack(anchor='w')
                ttk.Label(label_frame, text=value, font=('Segoe UI', 10, 'bold')).pack(anchor='w')
        except Exception as e:
            print(f"Error updating stats: {str(e)}")

    def update_person_dropdown(self):
        try:
            people = []
            if os.path.isdir('dataset/train'):
                for dir_name in os.listdir('dataset/train'):
                    if os.path.isdir(os.path.join('dataset/train', dir_name)):
                        people.append(dir_name)
            
            self.person_dropdown['values'] = people
            if people and not self.person_var.get() in people:
                self.person_dropdown.current(0)
        except Exception as e:
            print(f"Error updating dropdown: {str(e)}")

    def preview_image(self, img_path):
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((300, 300), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            
            self.preview_img = img_tk  # Keep reference to prevent garbage collection
            self.preview_canvas.create_image(150, 150, image=img_tk)
            
            # Try to extract face for better visualization
            face = self.extract_face(img_path)
            if face is not None:
                face_img = Image.fromarray(face)
                self.preview_label.config(text="Face detected ✅")
            else:
                self.preview_label.config(text="⚠️ No face detected")
        except Exception as e:
            print(f"Error previewing image: {str(e)}")
            self.preview_label.config(text=f"Error: {str(e)}")

    def select_test_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if file_path:
            self.test_image_path = file_path
            img = Image.open(file_path).convert('RGB')
            img = img.resize((300, 300), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            
            self.test_img = img_tk  # Keep reference
            self.test_canvas.create_image(150, 150, image=img_tk)
            
            self.run_test_btn.config(state='normal')

    def run_recognition_test(self):
        if not hasattr(self, 'test_image_path'):
            messagebox.showerror("Error", "Please select a test image first!")
            return
        
        try:
            # Load the trained model
            if not os.path.isfile(self.model_path):
                messagebox.showerror("Error", "No trained model found. Please train a model first!")
                return
            
            model = joblib.load(self.model_path)
            in_encoder = joblib.load(self.normalizer_path)
            out_encoder = joblib.load(self.encoder_path)
            
            # Extract face
            face = self.extract_face(self.test_image_path)
            if face is None:
                messagebox.showerror("Error", "No face detected in the image!")
                return
            
            # Get face embedding
            face_embedding = self.get_embedding(face)
            
            # Normalize embedding - ensure it's 2D
            face_embedding_reshaped = face_embedding.reshape(1, -1)
            face_embedding_norm = in_encoder.transform(face_embedding_reshaped)
            
            # Predict
            yhat_class = model.predict(face_embedding_norm)
            yhat_prob = model.predict_proba(face_embedding_norm)
            
            class_index = yhat_class[0]
            class_probability = yhat_prob[0, class_index] * 100
            
            # Get predicted name, handling potential errors
            try:
                predicted_name = out_encoder.inverse_transform([class_index])[0]
                # Clean any directory prefixes
                if isinstance(predicted_name, str) and '/' in predicted_name:
                    predicted_name = predicted_name.split('/')[-1]
            except Exception as e:
                print(f"Error transforming class index {class_index}: {str(e)}")
                predicted_name = f"Unknown (Class {class_index})"
            
            # Display results
            selected_person = self.person_var.get()
            # Clean any directory prefixes in selected person too
            if isinstance(selected_person, str) and '/' in selected_person:
                selected_person = selected_person.split('/')[-1]
                
            is_correct = selected_person == predicted_name
            
            self.results_text.config(state='normal')
            self.results_text.delete('1.0', tk.END)
            
            self.results_text.insert('1.0', f"Recognition Results:\n\n")
            self.results_text.insert(tk.END, f"Predicted Identity: {predicted_name}\n")
            self.results_text.insert(tk.END, f"Confidence: {class_probability:.2f}%\n\n")
            
            if selected_person:
                match_text = "✅ MATCH!" if is_correct else "❌ NO MATCH"
                self.results_text.insert(tk.END, f"Selected Person: {selected_person}\n")
                self.results_text.insert(tk.END, f"Result: {match_text}")
            
            self.results_text.config(state='disabled')
            
            # Draw the face box on the image
            self.draw_face_box_on_test_image()
            
        except Exception as e:
            messagebox.showerror("Error", f"Recognition failed: {str(e)}")
            print(f"Recognition error: {str(e)}")
            import traceback
            traceback.print_exc()

    def draw_face_box_on_test_image(self):
        """Draw a box around the detected face in the test image"""
        try:
            img = Image.open(self.test_image_path).convert('RGB')
            pixels = asarray(img)
            results = self.detector.detect_faces(pixels)
            
            if not results:
                return
                
            # Get the image for canvas
            img_display = img.resize((300, 300), Image.LANCZOS)
            img_width, img_height = img_display.size
            
            # Scale factors for the resized image
            orig_width, orig_height = img.size
            scale_x = img_width / orig_width
            scale_y = img_height / orig_height
            
            # Create PhotoImage
            self.test_img = ImageTk.PhotoImage(img_display)
            self.test_canvas.delete("all")
            self.test_canvas.create_image(150, 150, image=self.test_img)
            
            # Draw rectangle for each face, scaled to the display size
            for face in results:
                x1, y1, width, height = face['box']
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + width, y1 + height
                
                # Scale coordinates to match display size
                x1_scaled, y1_scaled = int(x1 * scale_x), int(y1 * scale_y)
                x2_scaled, y2_scaled = int(x2 * scale_x), int(y2 * scale_y)
                
                # Draw rectangle
                self.test_canvas.create_rectangle(
                    x1_scaled, y1_scaled, x2_scaled, y2_scaled,
                    outline=self.colors['accent'], width=2
                )
        except Exception as e:
            print(f"Error drawing face box: {str(e)}")

    def load_existing_models(self):
        """Load SVM model, label encoder, and normalizer if they exist with better error handling"""
        try:
            # First make sure paths are initialized
            if not hasattr(self, 'model_path'):
                self.initialize_paths()

            # Load SVM model
            if os.path.exists(self.model_path):
                print(f"Loading model from: {self.model_path}")
                try:
                    self.svm_model = joblib.load(self.model_path)
                except Exception as e:
                    print(f"Error loading SVM model: {e}")
                    self.svm_model = None
            else:
                print(f"No model found at: {self.model_path}")
                self.svm_model = None

            # Load label encoder
            if os.path.exists(self.encoder_path):
                print(f"Loading encoder from: {self.encoder_path}")
                try:
                    self.label_encoder = joblib.load(self.encoder_path)
                    print(f"Loaded label encoder with classes: {self.label_encoder.classes_}")
                except Exception as e:
                    print(f"Error loading label encoder: {e}")
                    self.label_encoder = LabelEncoder()
            else:
                print(f"No encoder found at: {self.encoder_path}")
                self.label_encoder = LabelEncoder()

            # Load normalizer
            if os.path.exists(self.normalizer_path):
                print(f"Loading normalizer from: {self.normalizer_path}")
                try:
                    self.in_encoder = joblib.load(self.normalizer_path)
                except Exception as e:
                    print(f"Error loading normalizer: {e}")
                    self.in_encoder = Normalizer(norm='l2')
            else:
                print(f"No normalizer found at: {self.normalizer_path}")
                self.in_encoder = Normalizer(norm='l2')

        except Exception as e:
            print(f"Error in load_existing_models: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Set defaults if loading fails
            self.svm_model = None
            self.label_encoder = LabelEncoder()
            self.in_encoder = Normalizer(norm='l2')


            


# Main application
if __name__ == "__main__":
    root = tk.Tk()
    app = ModernFaceGUI(root)
    root.mainloop()