# Neural_Network
Neural network from scratches in python

This neural network model uses the famous MNIST fashion dataset as the training data to classify different clothing fashion. 

*How to use:
  - First, download the MNIST fashion dataset and run the training.py, you can set the paramaters to any values that you want.
    (Such as learning_rate, decay, L2_bias_regularizer, L2_weight_regularizer,...)
  - After that, the training.py will automatically saves the model in 'trained.model' and its parameters in 'model.params' files. 
    All you need to do now is run the interface.py and select the image you want to classify.
    
*Note: 
  - This is my first neural network program, so there may still be some errors present. I have experimented with various parameter values to rectify any issues, 
    but the model may still make incorrect predictions, so feel free to adjust the code or the parameter values to improve the results. 🤪
    
*Example:
  - For example, in the training.py you can modify the values in these lines:
      "model.add(Layer_Dense(X.shape[1], 128, L2_weight_regularizer= 5e-4, L2_bias_regularizer= 5e-4))" tune the L2_weight_regularizer to 6e-7 or aything you want.
      "optimizer = Optimizer_Adam(learning_rate=0.005, decay=4e-3)" change the decay, learning_rate to any values that you want like 0.0004 or anything.
  - Or you can choose any other optimizer, loss and accuracy functions, you can find them in the neural_classes.py.

  
      
