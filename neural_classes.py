import numpy as np

data_labels = {
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

"""--------------------this part is for calculating accuracy--------------------"""
class Accuracy:
    def calculate(self, predictions, y):
        comparisons = self.compare(predictions, y)
        accuracy = np.mean(comparisons)

        #accumulated sum of sample count values after each batch of data
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)
        return accuracy

    def calculate_accumulated(self):
        accuracy = self.accumulated_sum / self.accumulated_count
        return accuracy
    
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

class Accuracy_Regression(Accuracy):
    def __init__(self):
        self.precision = None

    #calculate precisions value
    def initialize(self, y, reinitialize = False):
        if self.precision is None or reinitialize:
            #set a precision
            self.precision = np.std(y) / 250

    #compares predictions to the truth values:
    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision


#accuracy for classification model
class Accuracy_Categorical(Accuracy):
    def initialize(self, y):
        pass
    
    #compares prediction to the truth values
    def compare(self, predictions, y):
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)

        return predictions == y


"""--------------------this part for calculating loss--------------------"""
#calculate average loss
class Loss:
     #remember trainable layers
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def calculate(self, output, y, *, include_regularization = False):
        #calculate sample loss
        sample_losses = self.forward(output, y)
        #calculate mean loss
        data_loss = np.mean(sample_losses)

        #add accumulated sum of losses and sample count after each batch of data
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        #in validation data, we just need accuracy and loss, we don't need regularization_loss L1, L2
        #calculate loss in validation data
        if not include_regularization:
            return data_loss
        #return loss
        return data_loss, self.regularization_loss()

    def regularization_loss(self):
            #set to 0 by default
        regularization_loss = 0
        #calculate L1 - weights
        for layer in self.trainable_layers:
            if layer.weight_regularizer_L1 > 0: 
                regularization_loss += layer.weight_regularizer_L1 * np.sum(np.abs(layer.weights))

            #calculate L2 - weights
            if layer.weight_regularizer_L2 > 0:
                regularization_loss += layer.weight_regularizer_L2 * np.sum(layer.weights * layer.weights)

            #calculate L1 - biases
            if layer.bias_regularizer_L1 > 0:
                regularization_loss += layer.bias_regularizer_L1 * np.sum(np.abs(layer.biases))

            #L2 - biases
            if layer.bias_regularizer_L2 > 0:
                regularization_loss += layer.bias_regularizer_L2 * np.sum(np.abs(layer.biases * layer.biases))
                
        return regularization_loss

    def calculate_accumulated(self, *, include_regularization=False):
        data_loss = self.accumulated_sum / self.accumulated_count

        if not include_regularization:
            return data_loss

        return data_loss, self.regularization_loss()

    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

class Loss_CategoricalCrosssentropy(Loss):

    def forward(self, y_predict, y_true):
        #number of samples
        samples = len(y_predict)

        #clip data to prevent division by 0
        #clip both sides to not drag mean towards any value
        y_predict_clipped = np.clip(y_predict, 1e-7, 1 - 1e-7)

        #Probabilities for taget values only if categorical labels
        if len(y_true.shape) == 1: #nếu input (ở đây là softmax) là 1 list thì nhân với vector ví dụ (1, 0, 0)
            correct_confidences = y_predict_clipped[
                range(samples),
                y_true
            ]

        #mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2: #nếu là list trong list thì nhân với 1 set vector (one-hot encoded labels)
            correct_confidences = np.sum(
                y_predict_clipped * y_true,
                axis = 1 #sum of a row
            )

        #losses
        negative_log = -np.log(correct_confidences)
        return negative_log

    def backward(self, dvalues, y_true):

        samples = len(dvalues)
        #use the first sample to count them
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        #calculate the gradient
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples

class Loss_Binary_Crossentropy(Loss):
    def forward(self, y_predict, y_true):
        #clip data to prevent division by 0
        #works the same like Loss_CategoricalCrosssentropy
        y_predict_clipped = np.clip(y_predict, 1e-7, 1 - 1e-7)

        #calculate sample loss
        sample_losses = -(y_true * np.log(y_predict_clipped) + (1 - y_true) * np.log(1 - y_predict_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)

        return sample_losses

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])

        #clip data like above
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

        #calculate gradients
        self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs
        
        #normalize gradients
        self.dinputs = self.dinputs / samples

class Loss_MeanSquaredError(Loss): #L2
    def forward(self, y_predict, y_true):
        #calculate loss
        sample_losses = np.mean((y_true - y_predict)**2, axis=-1)
        return sample_losses

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        #use the first sample to count
        outputs = len(dvalues[0])

        self.dinputs = -2 * (y_true - dvalues) / outputs
        #normalize
        self.dinputs = self.dinputs / samples

class Mean_AbsoluteError(Loss):#L1
    def forward(self, y_predict, y_true):
        sample_losses = np.mean(np.abs(y_true - y_predict), axis=-1)
        return sample_losses

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        self.dinputs = np.sign(y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples


"""--------------------this part is for optimization classes--------------------"""
class Optimizer_SGD:
    def __init__(self, learning_rate=1.0, decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate #tracking the current learning rate
        self.decay = decay
        self.momentum = momentum
        self.iterations = 0 #step
    
    #decreases the learning rate per epoch during training
    def pre_update_parameters(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1 / (1 + self.decay * self.iterations))

    def update_parameters(self, layer):
        #using Stochastic Gradient Descent with momentum
        if self.momentum:
            #if layer does no contain momentum arrays then create them filled with zeros
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                #if there's no momentum arrays for weight then so does for biases
                layer.bias_momentums = np.zeros_like(layer.biases)

            #weight updates
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            #biases updates
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates

        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases
        
        #update weights and biases
        layer.weights += weight_updates
        layer.biases += bias_updates

    #after each step, the iterations will be added by 1
    def post_update_parameter(self):
        self.iterations += 1

class Optimizer_AdaGrad:
    #almost the same as Optimizer SGD
    def __init__(self, learning_rate=1.0, decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate #tracking the current learning rate
        self.decay = decay
        self.epsilon = epsilon
        self.iterations = 0 #step
    
    def pre_update_parameters(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1 / (1 + self.decay * self.iterations))

    def update_parameters(self, layer):
        #if layer doesn't contain cache arrays then create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        #update cache with squared current gradients
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2
        
        #cache += parameter_gradient**2
        #parameter_updates = learning_rate * parameter_gradient / (√(cache) + epsilon)
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_parameter(self):
        self.iterations += 1

class Optimizer_RMSprop:
    #initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate #tracking the current learning rate
        self.decay = decay
        self.epsilon = epsilon
        self.rho = rho
        self.iterations = 0 #step
    
    def pre_update_parameters(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1 / (1 + self.decay * self.iterations))

    def update_parameters(self, layer):
        #if layer doesn't contain cache arrays then create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        #update cache with squared current gradients
        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases**2
        
        #cache += parameter_gradient**2
        #parameter_updates = learning_rate * parameter_gradient / (√(cache) + epsilon)
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_parameter(self):
        self.iterations += 1 

class Optimizer_Adam: #recommended
    #initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate #tracking the current learning rate
        self.decay = decay
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.iterations = 0
        
    def pre_update_parameters(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1 / (1 + self.decay * self.iterations))

    def update_parameters(self, layer):
        #if layer doesn't contain cache arrays then create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)

        #update momentum with current gradients
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
        
        #corrected momentum
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        #update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights ** 2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases** 2

        #corrected cache
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        #SGD parameter update with normalization with square root cache
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_parameters(self):
        self.iterations += 1 


"""--------------------this part is for activation classes--------------------"""
#if the input is < 0 then it will take 0 as the output, otherwise it will take that input
class Activation_ReLU:
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        #copy the values
        self.dinputs = dvalues.copy()

        #zerp gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0

    def predictions(self, outputs):
        return outputs

#Softmax
class SoftMax_Activation:
    def forward(self, inputs, training): 
        #remember input values
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims= True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities#predicted values

    def backward(self, dvalues):
        #create uninitialized array
        self.dinputs = np.empty_like(dvalues)

        #enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            #calculate jacobian matrix of the output and sample-wise gradient
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)

class Activation_Softmax_Loss_CategoricalCrossentropy():
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        #if labels are one-hot encoded then turn them to discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs/samples

class Activation_Sigmoid:
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs)) # Sigmoid function: y = 1/(1 + e^(-x)) => y = [1 + e^(-x)]^(-1)

    def backward(self, dvalues):
        #derivatives
        self.dinputs = dvalues * (1 - self.output) * self.output # y' = y(1-y)

    def predictions(self, outputs):
        return (outputs > 0.5) * 1

class Linear_Activation():
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = inputs 

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

    def predictions(self, outputs):
        return outputs

"""--------------------this part is for layers--------------------"""
class Layer_Dense:
    #adding L1 and L2 regularization
    #sometimes if the model is stuck and does not learn at all
    #we should consider changing the factor in the weights initialization
    def __init__(self, inputs, neurons, L1_weight_regularizer = 0, L1_bias_regularizer = 0, L2_weight_regularizer = 0, L2_bias_regularizer = 0):
        self.weights = 0.01 * np.random.randn(inputs, neurons)
        self.biases = np.zeros((1, neurons))
        self.weight_regularizer_L1 = L1_weight_regularizer
        self.bias_regularizer_L1 = L1_bias_regularizer
        self.weight_regularizer_L2 = L2_weight_regularizer
        self.bias_regularizer_L2 = L2_bias_regularizer

    def forward(self, inputs, training):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs

    def backward(self, dvalues):
        #gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        if self.weight_regularizer_L1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_L1 * dL1

        if self.weight_regularizer_L2 > 0:
            self.dweights += 2 * self.weight_regularizer_L2 * self.weights

        if self.bias_regularizer_L1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_L1 * dL1

        if self.bias_regularizer_L2 > 0:
            self.dbiases += 2 * self.bias_regularizer_L2 * self.biases
        #gradients on values
        self.dinputs = np.dot(dvalues, self.weights.T)

    def retrieve_parameters(self):
        return self.weights, self.biases

    #set the parameters
    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases

class Layer_Dropout:
    def __init__(self, rate):
        #drop rate
        self.rate = 1 - rate

    def forward(self, inputs, training):
        self.inputs = inputs

        if not training:
            self.output = inputs.copy()
            return

        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask
    
    def backward(self, dvalues):
        #gradient on values
        self.dinputs = dvalues * self.binary_mask

class Layer_Input:
    def forward(self, inputs, training):
        self.output = inputs

