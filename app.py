from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load trained model
model = load_model("mask_detector.h5")

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Route: Home
@app.route('/')
def index():
    return render_template('index.html')

# Route: Prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Preprocess Image
    img = image.load_img(filepath, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        label = "🚫 No Mask Detected"
        confidence = round(float(prediction) * 100, 2)
    else:
        label = "😷 Mask Detected"
        confidence = round((1 - float(prediction)) * 100, 2)

    return render_template('result.html',
                           label=label,
                           confidence=confidence,
                           img_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)
