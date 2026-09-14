import os
import json
import cv2
import numpy as np
import pickle
from datetime import datetime
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

IMG_SIZE = 64
MODEL_PATH = "model/smile_model.pkl"
TRAINING_INFO_PATH = "model/training_info.json"
MAX_FILE_SIZE = 5000 * 1024  # 5000 KB


def convert_and_save_as_jpg(file_bytes, save_path, target_size=(IMG_SIZE, IMG_SIZE)):
    """Decode uploaded image bytes, resize to 64x64, save as a real .jpg file."""
    if len(file_bytes) > MAX_FILE_SIZE:
        raise ValueError("Image size exceeds 5000 KB limit")

    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Uploaded file is not a valid image")

    resized = cv2.resize(img, target_size)
    cv2.imwrite(save_path, resized)
    return save_path


def load_images_from_folder(folder_path, label):
    """Load every image in a folder and convert to grayscale feature vectors."""
    features = []
    labels = []
    if not os.path.exists(folder_path):
        return features, labels
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        features.append(resized.flatten())
        labels.append(label)
    return features, labels


def train_and_save_model(smile_dir, non_smile_dir, model_path=MODEL_PATH):
    """Train an SVM smile classifier on the given folders and save it as pickle."""
    smile_features, smile_labels = load_images_from_folder(smile_dir, 1)
    non_smile_features, non_smile_labels = load_images_from_folder(non_smile_dir, 0)

    X = smile_features + non_smile_features
    y = smile_labels + non_smile_labels

    if len(X) < 2:
        raise ValueError("Not enough images to train. Upload more images.")

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = SVC(kernel="linear")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return accuracy, len(X)


def predict_image(image_path, model_path=MODEL_PATH):
    """Load the saved model and predict smiling / not_smiling for one image."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    features = resized.flatten().reshape(1, -1)

    prediction = model.predict(features)[0]
    return "smiling" if prediction == 1 else "not_smiling"


def clear_folder(folder_path):
    """Delete every file inside a folder, then remove the folder itself."""
    if not os.path.exists(folder_path):
        return
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    os.rmdir(folder_path)


def save_training_info(accuracy, image_count):
    """Save the latest training run's stats: accuracy, image count, timestamp."""
    info = {
        "accuracy": round(accuracy * 100, 2),
        "image_count": image_count,
        "trained_at": datetime.now().strftime("%Y-%m-%d %I:%M %p"),
    }
    with open(TRAINING_INFO_PATH, "w") as f:
        json.dump(info, f)


def load_training_info():
    """Load the latest training run's stats, if any exist."""
    if not os.path.exists(TRAINING_INFO_PATH):
        return None
    with open(TRAINING_INFO_PATH, "r") as f:
        return json.load(f)