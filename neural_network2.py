import numpy as np
import pandas as pd
import code

class Dense:
    def __init__(self, input_dim, output_dim, activation_function):
        self.synapse = 2 * np.random.random((input_dim, output_dim)) - 1 
        self.select_activation_function(activation_function)
        
    def tanh(self, x):
        return np.tanh(x)

    def dtanh(self, y):
        return 1 - y ** 2

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def dsigmoid(self, y):
        return y*(1-y)
    
    def select_activation_function(self, activation_function):
        if activation_function == "tanh":
            self.activation_function = self.tanh
            self.activation_derivative = self.dtanh
        if activation_function == "sigmoid":
            self.activation_function = self.sigmoid
            self.activation_derivative = self.dsigmoid

    def forward(self, previous_layer):
        #code.interact(local=locals())
        self.output = self.activation_function(
            previous_layer.dot(self.synapse)
        )
        return self.output

    def compute_gradient(self, layer, error):
        self.delta = error * self.activation_derivative(layer)
        return self.delta.dot(self.synapse.T)
    
    def update_weights(self, layer, learning_rate):
        self.synapse += layer.T.dot(self.delta) * learning_rate

#multiple hidden layers
#np.array([1,2,3]).dot(self.synapse).dot(np.array(list(range(400))).reshape((20, 20)))
#single hidden layer
#np.array([1,2,3]).dot(self.synapse).dot(np.array(list(range(20))).reshape((20, 1)))

class Network:
    def __init__(self, layers):
        self.nn = layers
        self.layers = []

    def combine_input_and_synapse(self, X):
        output = X.dot(self.nn[0].synapse)
        output = self.nn[0].activation_function(output)
        self.layers.append(output)
        
    def forward(self, X):
        #self.combine_input_and_synapse(X)
        self.layers.append(X)
        for index, synapse in enumerate(self.nn):
            output = synapse.forward(self.layers[index])
            self.layers.append(output)
        return output

    def backpropagate(self, error, learning_rate):
        for index, layer in enumerate(reversed(self.layers)):
            error = self.nn[index].compute_gradient(layer, error)

        for index, synapse in enumerate(self.nn):
            synapse.update_weights(self.layers[index], learning_rate)

            
class NeuralNetwork:
    def __init__(self, learning_rate=0.1, target_mse=0.01, epochs=500):
        self.layers = []
        self.network = None
        self.learning_rate = learning_rate
        self.target_mse = target_mse
        self.epochs = epochs
        self.errors = []
        
    def add_layer(self, layer):
        self.layers.append(layer)

    def init_network(self):
        self.network = Network(self.layers)

    def fit(self, X, y):
        self.init_network()
        for epoch in range(self.epochs):
            self.errors = []

            rows, columns = X.shape
            for index in range(rows):
                # Forward
                output = self.network.forward(X[index])

                # Compute the error
                error = y[index] - output
                self.errors.append(error)

                # Back-propagate the error
                self.network.backpropagate(error, self.learning_rate)

            mse = (np.array(self.total_errors) ** 2).mean()
            if mse <= target_mse:
                break

    def predict(self, X):
        return self.network.forward(X)

def data_generation_easy():
    df = pd.DataFrame()
    for _ in range(1000):
        a = np.random.normal(0, 1)
        b = np.random.normal(0, 3)
        c = np.random.normal(12, 4)
        if a + b + c > 11:
            target = 1
        else:
            target = 0
        df = df.append({
            "A": a,
            "B": b,
            "C": c,
            "target": target
        }, ignore_index=True)
    return df

df = data_generation_easy()
column_names = ["A", "B", "C"]
target_name = "target"
X = df[column_names].values
y = np.array([[elem] for elem in list(df["target"].values)])

nn = NeuralNetwork()
# Dense(inputs, outputs, activation)
nn.add_layer(Dense(3, 20, "tanh"))
nn.add_layer(Dense(20, 1, "tanh"))
nn.fit(X, y)
y_pred = nn.predict(X)
print(mean_squared_error(y_pred, y))
