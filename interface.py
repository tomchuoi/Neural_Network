from model import Model, ImageClassifierUI

if __name__ == "__main__":
    # Load the trained model
    model = Model.load('trained.model')

    # Create an instance of the ImageClassifierUI class
    image_classifier_ui = ImageClassifierUI(model)

    # Start the main loop
    image_classifier_ui.root.mainloop()

# Start the main loop
root.mainloop()
