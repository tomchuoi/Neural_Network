from neural_classes import *
from colorama import Fore, Style
import pickle
import copy

#This model combines everything from classes :v
class Model:
    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            model = pickle.load(f)

        return model

    def __init__(self):
        #create a list of network
        self.layers = []
        self.softmax_output = None 

    #add object to the model
    def add(self, layer):
        self.layers.append(layer)
    
    #set loss and optimizer
    def set(self, *, loss = None, optimizer = None, accuracy = None):
        if loss is not None:
            self.loss = loss

        if optimizer is not None:
            self.optimizer = optimizer

        if accuracy is not None:
            self.accuracy = accuracy

    #train the model
    def train(self, X, y, *, epochs=1, batch_size = None, print_every=1, validation_data = None):
        train_steps = 1
        #initialize accuracy object
        self.accuracy.initialize(y)

        if batch_size is not None:
            train_steps = len(X) // batch_size
            if train_steps * batch_size < len(X):
                train_steps += 1

        #main training loop
        for epoch in range(1, epochs+1):
            print(f"──────────────────────────────────────────────────────────────────────────────────────────────────────────────\n" +
                  f"Epoch: {epoch}")

            #reset accumulated values in loss and accuracy objects
            self.loss.new_pass()
            self.accuracy.new_pass()

            for step in range(train_steps):
                if batch_size is None:
                    batch_X = X
                    batch_y = y

                else:
                    batch_X = X[step * batch_size:(step+1) * batch_size]
                    batch_y = y[step * batch_size:(step+1) * batch_size]
                
                output = self.forward(batch_X, training=True)
                #calculate loss
                data_loss, regularization_loss = self.loss.calculate(output, batch_y, include_regularization = True)
                loss = data_loss + regularization_loss

                #get predictions and calculate the accuracy
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)

                #perform backward pass
                self.backward(output, batch_y)

                #optimize
                self.optimizer.pre_update_parameters()
                for layer in self.trainable_layers:
                    self.optimizer.update_parameters(layer)
                self.optimizer.post_update_parameters()

                #print the output
                if not step % print_every or step == train_steps - 1:
                    print(f"Step: {step}| " +
                        f"Accuracy: {accuracy:.3f}| " +
                        f"Loss: {loss:.3f}| " +
                        f"Data_loss: {data_loss:.3f}| " +
                        f"Reg_loss {regularization_loss:.3f} " +
                        f"Current learning rate: {self.optimizer.current_learning_rate:.6f}")   

            epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(include_regularization = True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()

            print(f"{Fore.LIGHTGREEN_EX}[Training]{Style.RESET_ALL} " +
                        f"Accuracy: {accuracy:.3f}| " +
                        f"Loss: {loss:.3f}| " +
                        f"Data_loss: {data_loss:.3f}| " +
                        f"Reg_loss {regularization_loss:.3f} " +
                        f"Current learning rate: {self.optimizer.current_learning_rate:.6f}")   
            
            #validation data
            if validation_data is not None:
                self.evaluate(*validation_data, batch_size=batch_size) # * unpacks the validation data into singular value

    def finalize(self):
        self.input_layer = Layer_Input()
        layer_count = len(self.layers)
        self.trainable_layers = []

        #iterate the objects
        for i in range(layer_count):
            #if it's the first layer then the previous layer is the input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]

            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]

            #if it's the last layer then the next one is the loss
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            #if that layer has an attribute called 'weights' then it's trainable
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

            #update loss with trainable layers
            if self.loss is not None:        
                self.loss.remember_trainable_layers(self.trainable_layers)

        #if activation is softmax and loss is categorical cross entropy then combined them in an object
        if isinstance(self.layers[-1], SoftMax_Activation) and isinstance(self.loss, Loss_CategoricalCrosssentropy):
            self.softmax_output = Activation_Softmax_Loss_CategoricalCrossentropy()

    #create a forward pass
    def forward(self, X, training):
        #forward the input layer through this
        self.input_layer.forward(X, training)

        #call method of every object in a chain
        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        return layer.output

    def backward(self, output, y):

        #prioritize using softmax
        if self.softmax_output is not None:
            self.softmax_output.backward(output, y)
            self.layers[-1].dinputs = self.softmax_output.dinputs

            #call backward method all objects in reversed
            for layer in reversed (self.layers[:-1]):
                layer.backward(layer.next.dinputs)

            return

        self.loss.backward(output, y)
        #call backward method going through all objects in reversed order and and passing dinputs
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    def evaluate(self, X_val, y_val, *, batch_size = None):
        validation_steps = 1

        if batch_size is not None:
            #counting steps for validation dat
            validation_steps = len(X_val) // batch_size
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1

        self.loss.new_pass()
        self.accuracy.new_pass()
                
        for step in range(validation_steps):
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val

            else:
                batch_X = X_val[step * batch_size:(step+1) * batch_size]
                batch_y = y_val[step * batch_size:(step+1) * batch_size]

            output = self.forward(batch_X, training=False)
            #calculate loss
            self.loss.calculate(output, batch_y)

                #get predictions and calculate the accuracy
            predictions = self.output_layer_activation.predictions(output)
            self.accuracy.calculate(predictions, batch_y)

        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()

        print(f"{Fore.RED}[Validation]{Style.RESET_ALL} " +
                f"Accuracy: {validation_accuracy:.3f}| " +
                f"Loss: {validation_loss:.3f}\n")

    def retrieve_parameters(self):
        parameters = []
        for layer in self.trainable_layers:
            parameters.append(layer.retrieve_parameters())

        return parameters

    def set_parameters(self, parameters):
        for parameter_set, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*parameter_set)

    #saves the parameters into a file
    def save_parameters(self, path):
        with open(path, 'wb') as f: #wb is write in binary
            pickle.dump(self.retrieve_parameters(), f)

    #loads the parameters from the file into set_parameters function
    def load_parameters(self, path):
        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))

    #saves the model
    def save_model(self, path):
        model = copy.deepcopy(self)

        #reset accumulated values in loss and accuracy
        self.loss.new_pass()
        self.accuracy.new_pass()

        #remove any data in the input layer and reset gradient
        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('dinputs', None)

        #for each layer, removes all the inputs, output and dinputs
        for layer in model.layers:
            for property in ['inputs', 'output', 'dinputs', 'dweights', 'dbiases']:
                layer.__dict__.pop(property, None)

        #open the binary-write file to save the model
        with open(path, 'wb') as f:
            pickle.dump(model, f)

    def predict(self, X, *, batch_size = None):
        prediction_steps = 1

        if batch_size is not None:
            #counting steps for validation dat
            prediction_steps = len(X) // batch_size
            if prediction_steps * batch_size < len(X):
                prediction_steps += 1

        output = []
            
        for step in range(prediction_steps):
            #if batch size is not set then train using one step and full dataset
            if batch_size is None:
                batch_X = X

            else:
                batch_X = X[step * batch_size:(step+1) * batch_size]

            batch_output = self.forward(batch_X, training=False)
            output.append(batch_output)

        return np.vstack(output)
