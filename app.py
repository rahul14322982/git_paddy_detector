from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load model once when server starts
model = tf.keras.models.load_model("paddy_disease_model.h5")

class_names = [
    "bacterial_leaf_blight",
    "bacterial_leaf_streak",
    "bacterial_panicle_blight",
    "blast",
    "brown_spot",
    "brown_heat",
    "downy_mildew",
    "hispa",
    "normal",
    "tungro"

]

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img = Image.open(file.stream).convert("RGB")
    img = img.resize((224, 224))

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    print(prediction)

    max_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    return jsonify({
        "prediction": class_names[max_index],
        "confidence": confidence * 100
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
