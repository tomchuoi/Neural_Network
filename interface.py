from model import Model
from image_classifier import ImageClassifierUI

if __name__ == "__main__":
    model = Model.load('trained.model')
    image_classifier_ui = ImageClassifierUI(model)
    image_classifier_ui.root.mainloop()

# Start the main loop
root.mainloop()
