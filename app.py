from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__, static_url_path='/static', static_folder='static')

# Load the saved model
model = load_model("fire_detection_model.keras")

def predict_image(filename):
    img = image.load_img(filename, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    result = model.predict(img_array)
    return result[0][0]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']
        if file:
            file_path = "uploads/" + file.filename
            file.save(file_path)

            # Make a prediction
            result = predict_image(file_path)

            # Render the result on the webpage
            return render_template('index.html', result=result, file_path=file_path)

    return render_template('index.html', result=None, file_path=None)

if __name__ == '__main__':
    app.run(debug=True)
