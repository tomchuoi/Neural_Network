import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from neural_classes import *

class ImageClassifierUI:
    def __init__(self, model):
        # Create the main window
        self.root = tk.Tk()
        self.root.title("Image Classifier")
        self.root.attributes("-fullscreen", True)

        # Store the model
        self.model = model

        # Create a button to load the image
        self.load_button = tk.Button(self.root, text="Load Image", command=self.load_image)
        self.load_button.pack(pady=10)

        # Create a label to display the prediction result
        self.result_label = tk.Label(self.root, text="")
        self.result_label.pack()

        # Create a label to display the loaded image
        self.image_label = tk.Label(self.root)
        self.image_label.pack(pady=10)

        # Start the main loop
        self.root.mainloop()

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            # Load the image
            original_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

            # Resize the image
            resized_image = 255 - cv2.resize(image, (28, 28))
            input_data = (resized_image.reshape(1, -1).astype(np.float32) - 127.5) / 127.5

            # Predict the image
            self.predict_image(input_data)

            # Display the loaded image
            self.display_image(rgb_image)

    def predict_image(self, image_data):
        confidences = self.model.predict(image_data)
        predictions = self.model.output_layer_activation.predictions(confidences)
        prediction = data_labels[predictions[0]]

        # Display the prediction
        self.result_label.config(text=f"Prediction: {prediction}")

    def display_image(self, image):
        # Convert the image array to PIL Image format
        pil_image = Image.fromarray(image)

        # Resize the image to fit the display
        pil_image = pil_image.resize((200, 200))

        # Create Tkinter-compatible image
        displayed_image = ImageTk.PhotoImage(pil_image)

        # Update the image label
        self.image_label.config(image=displayed_image)
        self.image_label.image = displayed_image


if __name__ == "__main__":
    ImageClassifierUI()

