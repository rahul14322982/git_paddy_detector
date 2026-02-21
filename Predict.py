from cog import BasePredictor, Input
import tensorflow as tf
import numpy as np
from PIL import Image

class Predictor(BasePredictor):

    def setup(self):
        # Load model once when container starts
        self.model = tf.keras.models.load_model("paddy_disease_model1.h5")

        # Define class labels (CHANGE THIS according to your model)
        self.classes = [
            "bacterial_leaf_blight",
            "bacterial_leaf_streak",
            "bacterial_panicle_blight",
            "blast",
            "brown_mildew",
            "dead_heart",
            "downy_mildew",
            "hispa",
            "normal",
            "Tungro"
        ]

    def predict(
        self,
        image: Image = Input(description="Upload paddy leaf image")
    ) -> str:

        # Resize (CHANGE size if your model uses different input size)
        img = image.resize((224, 224))

        # Convert to numpy array
        img = np.array(img)

        # Normalize (if you trained with /255.0)
        img = img / 255.0

        # Add batch dimension
        img = np.expand_dims(img, axis=0)

        # Predict
        prediction = self.model.predict(img)

        # Get highest probability class
        class_index = np.argmax(prediction)

        # Return class name
        return self.classes[class_index]