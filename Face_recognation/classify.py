import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from random import choice
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder, Normalizer
from PIL import Image
import cv2


def display_test_image(image, prediction_info=None):
    """
    Display the test image with prediction information
    
    Args:
        image: Test image as numpy array
        prediction_info: Dictionary with prediction details (optional)
    """
    plt.figure(figsize=(8, 6))
    
    # Display the image
    if len(image.shape) == 3:
        if image.shape[2] == 3:
            plt.imshow(image)
        elif image.shape[2] == 1:
            plt.imshow(image.squeeze(), cmap='gray')
    elif len(image.shape) == 2:
        plt.imshow(image, cmap='gray')
    
    # Create title with prediction information
    if prediction_info:
        accuracy_indicator = "✓ Correct" if prediction_info["predicted"] == prediction_info["actual"] else "✗ Incorrect"
        title = f"predict: {prediction_info['predicted']} ({prediction_info['confidence']:.3f})\nActual: {prediction_info['actual']} | {accuracy_indicator}"
    else:
        title = "Test Image"
    
    plt.title(title, fontsize=14, pad=20)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def load_image_from_path(image_path):
    """Load image from file path and return as numpy array"""
    try:
        if os.path.exists(image_path):
            # Try loading with PIL first
            img = Image.open(image_path)
            img_array = np.array(img)
            return img_array
        else:
            print(f"Image path does not exist: {image_path}")
            return None
    except Exception as e:
        print(f"Error loading image from {image_path}: {e}")
        return None


def string_to_image(data_string, shape=(160, 160, 3)):
    """Convert string representation to image array"""
    try:
        # If the string contains image data, try to parse it
        if isinstance(data_string, str):
            # Remove brackets and split by whitespace
            data_string = data_string.strip('[]')
            # Try to convert to float array
            values = [float(x) for x in data_string.split()]
            img_array = np.array(values).reshape(shape)
            # Normalize to 0-255 range if needed
            if img_array.max() <= 1.0:
                img_array = (img_array * 255).astype(np.uint8)
            return img_array
        return None
    except Exception as e:
        print(f"Error converting string to image: {e}")
        return None


def process_face_data(face_data):
    """
    Process face data regardless of its format (string, path, or array)
    
    Args:
        face_data: Face data in various formats
        
    Returns:
        numpy array of image or None if conversion fails
    """
    if face_data is None:
        return None
    
    # Check if it's already a numpy array
    if isinstance(face_data, np.ndarray):
        # If it's a 0-dimensional array (scalar), extract the value
        if face_data.ndim == 0:
            face_data = face_data.item()
        else:
            # Check if it's a valid image array
            if face_data.ndim >= 2:
                return face_data
    
    # If it's a string, try different approaches
    if isinstance(face_data, str):
        print(f"Processing string data: {face_data[:100]}...")  # Show first 100 chars
        
        # Check if it's a file path
        if os.path.exists(face_data):
            print("String appears to be a file path, loading image...")
            return load_image_from_path(face_data)
        
        # Check if it's serialized image data
        if face_data.startswith('[') or ' ' in face_data:
            print("String appears to contain image data, attempting conversion...")
            return string_to_image(face_data)
        
        # If it's a very long string, it might be base64 or other encoded data
        if len(face_data) > 1000:
            print("String is very long, might be encoded image data")
            # You could add base64 decoding here if needed
            return None
    
    print(f"Unable to process face data of type: {type(face_data)}")
    return None


# Check if dataset files exist
if not os.path.exists("dataset.npz") or not os.path.exists("embeddings.npz"):
    raise FileNotFoundError("Error: dataset.npz or embeddings.npz file not found!")

# Load face dataset (used for visualization)
data = np.load("dataset.npz", allow_pickle=True)
if "arr_2" not in data:
    raise KeyError("Error: dataset.npz does not contain 'arr_2' (testX_faces).")
testX_faces = data["arr_2"]

# Also try to load additional information if available
test_names = None
if "arr_3" in data:  # If dataset.npz contains names
    test_names = data["arr_3"]

# Check if face paths are available for better matching
face_paths_to_images = {}
if os.path.exists("image_mapping.npz"):
    try:
        mapping_data = np.load("image_mapping.npz", allow_pickle=True)
        if "paths" in mapping_data and "images" in mapping_data:
            paths = mapping_data["paths"]
            images = mapping_data["images"]
            for i in range(len(paths)):
                face_paths_to_images[paths[i]] = images[i]
        print("Loaded image mapping file for better face matching")
    except Exception as e:
        print(f"Note: Could not load image mapping: {e}")

# Load embeddings dataset
data = np.load("embeddings.npz", allow_pickle=True)
required_keys = ["arr_0", "arr_1", "arr_2", "arr_3"]
if not all(key in data for key in required_keys):
    raise KeyError("Error: embeddings.npz is missing one or more required arrays.")

# In the training code, these arrays represent:
# arr_0: Face embeddings (trainX)
# arr_1: Person names (trainy)
# arr_2: Encoded label numbers - not raw testX as the variable name suggests
# arr_3: Paths to face images - not testy as the variable name suggests

# Let's rename these variables to reflect their actual content
embeddings = data["arr_0"]  # Face embeddings
names = data["arr_1"]       # Person names (labels)
encoded_labels = data["arr_2"]  # Encoded labels
face_paths = data["arr_3"]  # Paths to images

# Check if embeddings is 1D and reshape if needed
if len(embeddings.shape) == 1:
    embeddings = embeddings.reshape(-1, 1)  # Reshape to 2D array if it's 1D

# Normalize embeddings
in_encoder = Normalizer(norm="l2")
embeddings = in_encoder.transform(embeddings)

# Load label encoder and use it directly
try:
    # Try to load the saved label encoder
    out_encoder = joblib.load("label_encoder.pkl")
except (FileNotFoundError, IOError):
    # If not available, create a new one and fit it to the names
    print("Warning: label_encoder.pkl not found. Creating a new encoder.")
    out_encoder = LabelEncoder()
    out_encoder.fit(names)
    
# Clean any paths from the names
cleaned_names = [name.replace('train/', '').replace('val/', '') if isinstance(name, str) else name for name in names]

# Re-encode the names with the loaded encoder
trainy = out_encoder.transform(cleaned_names)

# Load the trained model
model = joblib.load("svm_face_model.pkl")

# Select a random sample
selection = choice(range(embeddings.shape[0]))
random_face_emb = embeddings[selection]
random_face_class = trainy[selection]
random_face_name = out_encoder.inverse_transform([random_face_class])[0]

# Try to find the matching face image using improved logic
matched_face = None

# First, try the direct mapping approach
if len(face_paths) > selection and face_paths[selection] in face_paths_to_images:
    # If we have a direct path-to-image mapping
    matched_face = face_paths_to_images[face_paths[selection]]
    print(f"Found matching face image using image mapping")
elif len(face_paths) > selection:
    # Try to load from the path directly
    face_path = face_paths[selection]
    matched_face = load_image_from_path(face_path)
    if matched_face is not None:
        print(f"Loaded face image directly from path: {face_path}")

# If direct path didn't work, try finding by name in test dataset
if matched_face is None and test_names is not None:
    # Find faces with the same name
    matching_indices = [i for i, name in enumerate(test_names) if 
                      (isinstance(name, str) and random_face_name in name) or 
                      name == random_face_name]
    
    if matching_indices:
        # Use the first matching face
        matched_index = matching_indices[0]
        raw_face_data = testX_faces[matched_index]
        matched_face = process_face_data(raw_face_data)
        if matched_face is not None:
            print(f"Found matching face by name in test dataset")

# If still no match, use fallback selection
if matched_face is None:
    raw_face_data = testX_faces[selection % len(testX_faces)]
    matched_face = process_face_data(raw_face_data)
    if matched_face is not None:
        print(f"Using fallback face selection")

# Predict the face
samples = expand_dims(random_face_emb, axis=0)
yhat_class = model.predict(samples)
yhat_prob = model.predict_proba(samples)

# Get prediction details
class_index = yhat_class[0]
class_probability = yhat_prob[0, class_index]
predict_name = out_encoder.inverse_transform(yhat_class)[0]

# Print results
print(f"Predicted: {predict_name} ({class_probability:.3f})")
print(f"Actual: {random_face_name}")

# Display the test image with prediction results
if matched_face is not None:
    print(f"\nDisplaying test image...")
    print(f"Image shape: {matched_face.shape}")
    print(f"Image dtype: {matched_face.dtype}")
    
    # Ensure the image has proper data type and range
    if matched_face.dtype != np.uint8:
        if matched_face.max() <= 1.0:
            matched_face = (matched_face * 255).astype(np.uint8)
        else:
            matched_face = matched_face.astype(np.uint8)
    
    print(f"Image value range: [{matched_face.min()}, {matched_face.max()}]")
    
    # Prepare prediction information
    prediction_info = {
        'predicted': predict_name,
        'actual': random_face_name,
        'confidence': class_probability
    }
    
    # Display the test image
    display_test_image(matched_face, prediction_info)

else:
    print("No valid image data available for display")
    print("This might indicate an issue with how the image data is stored in your dataset files.")
    
    # Print some debugging information
    if len(testX_faces) > 0:
        sample_face = testX_faces[0]
        print(f"Sample face data type: {type(sample_face)}")
        if hasattr(sample_face, 'shape'):
            print(f"Sample face shape: {sample_face.shape}")
        if hasattr(sample_face, 'dtype'):
            print(f"Sample face dtype: {sample_face.dtype}")
        
        # Try to show what the data looks like
        if isinstance(sample_face, str):
            print(f"Sample face string (first 200 chars): {str(sample_face)[:200]}")
        elif isinstance(sample_face, np.ndarray) and sample_face.size < 10:
            print(f"Sample face array: {sample_face}")