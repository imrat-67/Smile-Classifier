import cv2
import numpy as np
import pickle

IMG_SIZE = 64
MODEL_PATH = "model/smile_model.pkl"

def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    flattened = resized.flatten()
    features = flattened.reshape(1, -1)
    return features

def predict_smile(image_path):
    model = load_model()
    features = preprocess_image(image_path)
    prediction = model.predict(features)[0]

    if prediction == 1:
        return "smiling"
    else:
        return "not_smiling"

if __name__ == "__main__":
    test_image = "dataset/test/Aaron_Pena_0001.jpg"
    result = predict_smile(test_image)
    print(f"Prediction: {result}")