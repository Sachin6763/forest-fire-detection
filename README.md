# Fire Detection Project

## Overview

This project focuses on detecting the presence of fire in images using a Convolutional Neural Network (CNN). The project includes both the model development code and a user-friendly web interface for interacting with the model.

### Features

- **Convolutional Neural Network (CNN):** Core of the project for identifying fire patterns in images.
- **Training and Testing Datasets:** Separate datasets for training and testing the model.
- **Web Interface:** Flask-based interface for easy image uploads and real-time predictions.

## Project Structure

- `test.py`: Vs Code for creating, compiling, and training the CNN model.
- `app.py`: Flask application script for handling image uploads, making predictions, and rendering the web interface.
- `templates/index.html`: HTML template for the web interface.
- `static/`: Directory containing static files (CSS, images).
- `uploads/`: Directory where uploaded images are temporarily stored.

## Getting Started

1. **Install Dependencies:** `pip install -r requirements.txt`
2. **Train the Model (Optional):** Run code in `test.py`.
3. **Run the Web Interface:** `python app.py`
4. **Access the Interface:** Open a web browser and go to `http://localhost:5000`

## Usage

1. Open the web interface and click on "Choose an image..."
2. Select an image file containing a scene with or without fire.
3. Click "Submit" to upload the image and receive a real-time prediction.
4. View the result indicating whether the uploaded image contains fire or not.

## Contributions

Contributions are welcome! Please follow GitHub practices for forking, creating branches, and submitting pull requests.
