from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from class_names import CLASS_NAMES

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL = tf.keras.models.load_model("../saved_models/1.keras")

def read_file_as_img(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.get('/')
async def home():
    return 'Hello! Welcome to the server'

@app.post('/predict')
async def predict(file: UploadFile = File(...)):

    image = read_file_as_img(await file.read())

    image_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(image_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return {
        'class': predicted_class,
        'confidence': float(confidence)
        }
