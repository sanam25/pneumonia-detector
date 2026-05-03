from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build model architecture (same as training)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=x)

# Load weights
model.load_weights("model.weights.h5")

# Preprocess image
def preprocess(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# Prediction API
@app.post("/predict")
@app.get("/")
def home():
    return {"message": "Pneumonia Detection API is running"}
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = preprocess(contents)

    pred = model.predict(img)[0][0]

    if pred > 0.5:
        result = "PNEUMONIA"
        confidence = float(pred)
    else:
        result = "NORMAL"
        confidence = float(1 - pred)

    # Extra features (makes your project better)
    if confidence > 0.8:
        severity = "HIGH"
        advice = "Consult a doctor immediately"
    elif confidence > 0.6:
        severity = "MODERATE"
        advice = "Medical check recommended"
    else:
        severity = "LOW"
        advice = "Normal condition"

    return {
        "prediction": result,
        "confidence": round(confidence * 100, 2),
        "severity": severity,
        "advice": advice
    }