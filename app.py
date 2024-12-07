from flask import Flask, request, jsonify
from skimage.util.shape import view_as_blocks
from skimage import io, transform
import tensorflow as tf
import numpy as np
from tensorflow import keras
import os
from tensorflow.keras.models import load_model
port=5000
# Set the environment variable to disable oneDNN optimizations
app = Flask(__name__)

# Function to convert one-hot encoded array to FEN notation
def fen_from_onehot(one_hot):
    piece_symbols = 'PNBRQKpnbrqk'
    output = ''
    for j in range(8):
        for i in range(8):
            if one_hot[j][i] == 12:
                output += ' '
            else:
                output += piece_symbols[one_hot[j][i]]
        if j != 7:
            output += '/'

    for i in range(8, 0, -1):
        output = output.replace(' ' * i, str(i))

    return output


# Function to process the image
def process_image(img_path):
    downsample_size = 200
    square_size = int(downsample_size / 8)
    img_read = io.imread(img_path)
    img_read = transform.resize(img_read, (downsample_size, downsample_size), mode='constant')
    tiles = view_as_blocks(img_read, block_shape=(square_size, square_size, 3))
    tiles = tiles.squeeze(axis=2)
    return tiles.reshape(64, square_size, square_size, 3)


# Load the trained model
modelf = load_model('recent_chess_trained_model.keras',compile=False)


# Function to predict FEN from image
def predict_fen(image_path):
    processed_image = process_image(image_path)
    pred = modelf.predict(processed_image).argmax(axis=1).reshape(-1, 8, 8)
    fen = fen_from_onehot(pred[0])
    return fen



# Endpoint to receive image file and return the predicted FEN
@app.route("/predict_fen", methods=["POST"])
def predict():
    """
    Endpoint to receive image file and return the predicted FEN as plain text.
    """
    try:
        # Get the uploaded image from the request
        image = request.files.get('image')

        if not image:
            return jsonify({"error": "No image file provided"}), 400

        # Save the image to a temporary location
        image_path = os.path.join('uploads', image.filename)
        image.save(image_path)
        print("recieved")
        # Get FEN prediction
        fen = predict_fen(image_path)
        print(fen)
        
        # Return the FEN as plain text
        return jsonify({"fen": fen}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Ensure the uploads directory exists
    os.makedirs('uploads', exist_ok=True)
    
    app.run(port)
