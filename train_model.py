import os
import cv2
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

IMG_SIZE = 64
DATASET_DIR = "dataset"
MODEL_PATH = "model/smile_model.pkl"

def load_images_from_folder(folder_path, label):
    features = []
    labels = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        flattened = resized.flatten()
        features.append(flattened)
        labels.append(label)
    return features, labels

def main():
    smile_features, smile_labels = load_images_from_folder(os.path.join(DATASET_DIR, "smile"), 1)
    non_smile_features, non_smile_labels = load_images_from_folder(os.path.join(DATASET_DIR, "non_smile"), 0)

    X = smile_features + non_smile_features
    y = smile_labels + non_smile_labels

    X = np.array(X)
    y = np.array(y)

    print(f"Total images: {len(X)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = SVC(kernel="linear")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc * 100:.2f}%")

    os.makedirs("model", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()