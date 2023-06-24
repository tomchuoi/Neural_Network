# Neural_Network
Neural network from scratches in python

This neural network model uses the famous MNIST fashion dataset as the training data to classify different clothing fashion. 

*How to use:
  - Start by running config.py to download the essential modules required for the program.
  - Next, download the MNIST fashion dataset. This dataset will be used for training the model. 
  - Customize parameters in training.py as needed and before running it(e.g., learning rate, decay, regularizers).
  - Run interface.py to classify images. Select the desired image in the interface.
    
*Note: 
  - This is my first neural network program, so there may still be some errors present. I have experimented with various parameter values to rectify any issues, 
    but the model may still make incorrect predictions, so feel free to adjust the code or the parameter values to improve the results. 🤪
    
*Example:
  - For example, in the training.py you can modify the values in these lines:
      "model.add(Layer_Dense(X.shape[1], 128, L2_weight_regularizer= 5e-4, L2_bias_regularizer= 5e-4))" tune the L2_weight_regularizer to 6e-7 or anything you want.
      "optimizer = Optimizer_Adam(learning_rate=0.005, decay=4e-3)" you can change the decay and learning_rate to any values you prefer, such as 0.0004 or any other value.
  - Or you can choose any other optimizer, loss and accuracy functions, you can find them in the neural_classes.py. 
  - Feel free to experiment with these parameters and functions to improve the performance of the model according to your needs and preferences.

  
      
