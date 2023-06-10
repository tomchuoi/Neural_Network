from neural_classes import *
from model import Model
from dataset import *
import os

# Delete the old training data and parameters
if os.path.exist('trained.model'):
    os.remove('trained.model')
if os.path.exist('model.params'):
    os.remove('model.params')
    
# Create dataset
X, y, X_test, y_test = create_dataset('fashion_mnist_images')

# Shuffle the training the dataset
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

# Scale and reshape
X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5

model = Model()

# Add layers
model.add(Layer_Dense(X.shape[1], 128, L2_weight_regularizer= 5e-4, L2_bias_regularizer= 5e-4))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.05))
model.add(Layer_Dense(128, 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 10))
model.add(SoftMax_Activation())

# Set the model
model.set(
    loss = Loss_CategoricalCrosssentropy(),
    optimizer = Optimizer_Adam(learning_rate=0.005, decay=4e-3),
    accuracy = Accuracy_Categorical()
)

model.finalize()

# Train the model
model.train(X, y, validation_data=(X_test, y_test), epochs=20, batch_size=128, print_every=100)

# Save the parameters
model.save_parameters('model.params')

# Save the trained model
model.save_model('trained.model')
