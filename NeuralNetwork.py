import numpy as np

class NeuralNetwork:

    def __init__(self,hidden_layers=2, lr=0.01, layer_neurons=8, n_iters=100, final_neurons=9):
        self.hidden_layers = hidden_layers
        self.lr = lr
        self.nuerons = layer_neurons
        self.bias = [
            np.zeros((self.nuerons, 1))
            for _ in range(self.hidden_layers)
        ]
        self.final_neurons = final_neurons
        self.bias.append(np.zeros((self.final_neurons, 1)))
        self.n_iters = n_iters
    
    def _init_weights(self, X):
        self.weights = [
            np.random.randn(self.nuerons, X.shape[1] if i == 0 else self.nuerons)
            * np.sqrt(2 / (X.shape[1] if i == 0 else self.nuerons))
            for i in range(self.hidden_layers)
        ]
        self.weights.append(
            np.random.randn(self.final_neurons, self.nuerons) * 0.01
        )

    def fit(self, X, y,multi = False):
        X = np.array(X, dtype=float)

        if multi:
            y = np.asarray(y)     
        else:
            y = np.asarray(y).reshape(-1)   

        self._init_weights(X)
        self.m = X.shape[0]

        for enum in range(self.n_iters):
            layers_data, z_data = self._forward(X)
            probs = self.softmax(layers_data[-1])
            self._backward(layers_data, z_data, probs, y, multi=multi)

            #if enum % 100 == 0:
                #y_onehot = self._one_hot(y)
                #probs_clipped = np.clip(probs, 1e-15, 1 - 1e-15)
                #loss = -np.sum(y_onehot * np.log(probs_clipped)) / self.m
                #print(enum, loss)

        return self
    
    def _forward(self,X):
        layers_data = [X.T]
        z_data = []

        for i in range(self.hidden_layers + 1):
            z = np.dot(self.weights[i], layers_data[i]) + self.bias[i]
            z_data.append(z)

            if i == self.hidden_layers:
                a = z  # no activation on output layer
            else:
                a = self.ReLU(z)

            layers_data.append(a)

        return layers_data, z_data
    
    def _backward(self,layers_data,z_data,probs,y,multi = False):

        if multi:
            y_onehot = self._one_hot_multi(y)
        else:
            y_onehot = self._one_hot(y)
            
        probs_clipped = np.clip(probs, 1e-15, 1.0 - 1e-15)
        ce = -np.sum(y_onehot * np.log(probs_clipped)) / self.m

        dZ_last = probs - y_onehot

        for i in reversed(range(self.hidden_layers + 1)):
            data = layers_data[i]
            w = self.weights[i]

            dW = np.dot(dZ_last, data.T) / self.m
            db = np.sum(dZ_last, axis=1, keepdims=True) / self.m
            dA = np.dot(w.T, dZ_last)

            if i != 0:  
                z = z_data[i - 1]
                relu_derivative = (z > 0).astype(float)
                dZ_last = dA * relu_derivative
                
            self.weights[i] = self.weights[i] - self.lr*(dW)
            self.bias[i] = self.bias[i] - self.lr*(db)
    
    def ReLU(self,x):
        return np.maximum(0,x)
    
    def _one_hot(self, y):
        m = len(y)
        y_onehot = np.zeros((self.final_neurons, m))
        y_onehot[y, np.arange(m)] = 1
        return y_onehot
    
    def _one_hot_multi(self, best_moves_all_col):
        m = len(best_moves_all_col)
        y_onehot = np.zeros((self.final_neurons, m))

        for j, cell in enumerate(best_moves_all_col):
            moves = [int(x) for x in str(cell).split(";")]
            y_onehot[moves, j] = 1.0 / len(moves)  

        return y_onehot
    
    def softmax(self, z):
        z_shifted = z - np.max(z, axis=0, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def _get_y_hat(self, probs):
        return np.argmax(probs, axis=0)

    def predict(self, X):
        X = np.array(X, dtype=float)       
        layers_data, _ = self._forward(X)
        probs = self.softmax(layers_data[-1])
        y_hat = self._get_y_hat(probs)
        return y_hat
    
    def save(self, path):
        
        np.savez(
            path,
            hidden_layers=self.hidden_layers,
            neurons=self.nuerons,
            final_neurons=self.final_neurons,
            **{f"W{i}": w for i, w in enumerate(self.weights)},
            **{f"b{i}": b for i, b in enumerate(self.bias)},
        )

    @classmethod
    def load(cls, path):
        data = np.load(path)
        model = cls(
            hidden_layers=int(data["hidden_layers"]),
            layer_neurons=int(data["neurons"]),
            final_neurons=int(data["final_neurons"]),
        )

        model.weights = [data[f"W{i}"] for i in range(model.hidden_layers + 1)]
        model.bias = [data[f"b{i}"] for i in range(model.hidden_layers + 1)]
        return model
