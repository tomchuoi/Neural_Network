import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from neural_classes import *
from model import Model
import cv2

# Load the trained model
model = Model.load('trained.model')

def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Load the image
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        # Resize the image
        resized_image = 255 - cv2.resize(image, (28, 28))
        input_data = (resized_image.reshape(1, -1).astype(np.float32) - 127.5) / 127.5

        # Predict the image
        predict_image(input_data)

        # Display the loaded image
        display_image(image)

def predict_image(image_data):
    confidences = model.predict(image_data)
    predictions = model.output_layer_activation.predictions(confidences)
    prediction = data_labels[predictions[0]]

    # Display the prediction
    result_label.config(text=f"Prediction: {prediction}")

def display_image(image):
    # Convert the image array to PIL Image format
    pil_image = Image.fromarray(image)

    # Resize the image to fit the display
    pil_image = pil_image.resize((200, 200))

    # Create Tkinter-compatible image
    tk_image = ImageTk.PhotoImage(pil_image)

    # Update the image label
    image_label.config(image=tk_image)
    image_label.image = tk_image

# Create the main window
root = tk.Tk()
root.title("Image Classifier")

# Create a button to load the image
load_button = tk.Button(root, text="Load Image", command=load_image)
load_button.pack(pady=10)

# Create a label to display the prediction result
result_label = tk.Label(root, text="")
result_label.pack()

# Create a label to display the loaded image
image_label = tk.Label(root)
image_label.pack(pady=10)

# Start the main loop
root.mainloop()
