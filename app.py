#pip install flask tensorflow pillow numpy

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os
import uuid

app = Flask(__name__)
model = load_model("best_mammal_model.h5")

# Class labels in the same order as your training data
class_labels = ['Bear', 'Cat', 'Dog', 'Elephant', 'Goat', 'Horse', 'Lion', 'Tiger', 'Wolf']

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_path = None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = str(uuid.uuid4()) + ".jpg"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            img = Image.open(filepath).convert('RGB')
            img = img.resize((224, 224))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            preds = model.predict(img_array)[0]
            predicted_label = class_labels[np.argmax(preds)]
            confidence = np.max(preds) * 100

            prediction = f"{predicted_label} ({confidence:.2f}%)"
            image_path = filepath

    return render_template('index.html', prediction=prediction, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
