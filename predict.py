from neural_classes import *
from model import Model
from dataset import *

data_labels= {
    0: 'Áo cộc tay',
    1: 'Quần dài',
    2: 'Áo len',
    3: 'Áo váy',
    4: 'Áo khoác',
    5: 'Sandal',
    6: 'Áo dài tay',
    7: 'Giày',
    8: 'Túi',
    9: 'Bốt'
}

# Read the image
image_data = cv2.imread('ohshiet.png', cv2.IMREAD_GRAYSCALE)

# Resize the image
image_data = cv2.resize(image_data, (28, 28))


image_data = 255 - image_data

# Reshape and scale the data
image_data = (image_data.reshape(1, -1).astype(np.float32) - 127.5) / 127.5

# Load the trained model
model = Model.load('trained.model')

confidences = model.predict(image_data)
predictions = model.output_layer_activation.predictions(confidences)
prediction = data_labels[predictions[0]]

print(prediction)

