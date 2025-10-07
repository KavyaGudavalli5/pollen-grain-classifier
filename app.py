from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Create Flask app
app = Flask(__name__)

# Classes
class_names = ['rose', 'sunflower', 'lily', 'hibiscus', 'lotus', 'lavender']

# Load dummy model
model = load_model("temp_model.h5")  # Make sure this file exists in the same folder

# Route
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    filename = None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Save uploaded file
            filepath = os.path.join('uploads', file.filename)
            file.save(filepath)
            filename = file.filename

            # Prepare image for prediction
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # Predict
            preds = model.predict(img_array)
            prediction = class_names[np.argmax(preds)]

    return render_template('index.html', prediction=prediction, filename=filename)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)