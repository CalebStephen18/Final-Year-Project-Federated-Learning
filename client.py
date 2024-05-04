import tensorflow as tf
import flwr as fl
import keras
from keras import layers
import pandas as pd
import os
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import Dense, Conv1D, Flatten, SimpleRNN
from keras.optimizers import Adam

model = keras.Sequential([
        layers.Input(shape=(7,)),
        layers.Reshape((7, 1)),  # Reshape input for 1D convolution
        SimpleRNN(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
])

# Compiling the model
custom_optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=custom_optimizer, loss='mean_squared_error')


class FlowerClient(fl.client.NumPyClient):

    def __init__(self, client_id, dataset_path):
        super().__init__()
        self.client_id = client_id
        self.dataset_path = dataset_path
        self.client_names = ["bang1", "bom1", "cbe1", "hyd1", "maa1", "mad1"]
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.THRESHOLD = 0.5
    def load_dataset(self):
        client_df = pd.read_csv(os.path.join(self.dataset_path, f"{self.client_names[self.client_id]}.csv"))
        X_train = client_df.iloc[:,:-1].values
        y_train = client_df.iloc[:,-1].values
        self.X_train, self.y_train = X_train, y_train
        client_df = None
    
    def release_dataset(self):
        self.X_train = None
        self.y_train = None
    
    def load_dataset_test(self):
        test_df = pd.read_csv(os.path.join(self.dataset_path, f"testset.csv")).values
        X_test = test_df.iloc[:,:-1].values
        y_test = test_df.iloc[:,-1].values
        self.X_train, self.y_train = X_test, y_test
    
    def release_dataset_test(self):
        self.X_train = self.y_train = None

    def fit(self, parameters, config):
        if self.X_train is None or self.y_train is None:
            self.load_dataset()

        custom_optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=custom_optimizer, loss='mean_squared_error')
        model.set_weights(parameters)
        model.fit(self.X_train, self.y_train, epochs=1, batch_size=32)

        dataset_len = len(self.X_train)

        # Release the dataset from memory
        self.release_dataset()

        self.client_id += 2
        if self.client_id >= 6:
            self.client_id = self.client_id % 6

        return model.get_weights(), dataset_len, {}

    def evaluate(self, parameters, config):
        if self.X_train is None or self.y_train is None:
            self.load_dataset()
        custom_optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=custom_optimizer, loss='mean_squared_error')
        model.set_weights(parameters)
        y_pred = model.predict(self.X_train)
        mse = mean_squared_error(self.y_train, y_pred)
        loss = model.evaluate(self.X_train, self.y_train)
        len_dataset = len(self.X_train)
        self.release_dataset()
        return loss, len_dataset, {"mse":mse}

    def get_parameters(self, config):
        return model.get_weights()

if __name__ == "__main__":
    fl.client.start_numpy_client(server_address="127.0.0.1:8087", client=FlowerClient(client_id=1, dataset_path="./datasets/version_0"))