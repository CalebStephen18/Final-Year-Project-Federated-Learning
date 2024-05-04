import flwr as fl
from flwr.common import Metrics
from tensorflow.keras.layers import Dense, Conv1D, Flatten, SimpleRNN
from typing import Dict, Optional, Tuple, List
import pandas as pd
import os
from sklearn.metrics import mean_squared_error
import keras
import tensorflow as tf
from keras import layers
from keras.optimizers import Adam

DATASET_PATH = "./datasetz"

def main():
    model = keras.Sequential([
        layers.Input(shape=(7,)),
        layers.Reshape((7, 1)),  
        SimpleRNN(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
])

    # Compiling the model
    custom_optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=custom_optimizer, loss='mean_squared_error')

    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(model),
        evaluate_metrics_aggregation_fn=weighted_average,
        min_fit_clients=2
    )

    num_clients = 6
    num_active_clients = 2
    num_epochs = 30

    num_rounds = (num_clients // num_active_clients) * num_epochs

    fl.server.start_server(
        server_address="0.0.0.0:8087",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy
    )

def save_model(model):
    model_dir = "./model_results_"
    os.makedirs(model_dir, exist_ok=True)
    model_round = len(os.listdir(model_dir))
    model_filename = f"custom_covnet_{model_round}.h5"
    # Save the model in HDF5 format
    model.save(os.path.join(model_dir, model_filename))

def load_dataset_test():
    test_df = pd.read_csv(os.path.join(DATASET_PATH, f"testset.csv"))
    X_test = test_df.iloc[:,:-1].values
    y_test = test_df.iloc[:,-1].values
    return X_test, y_test

def get_evaluate_fn(model):
    """Return an evaluation function for server-side evaluation."""

    x_val, y_val = load_dataset_test()

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(parameters)  # Update model with the latest parameters
        y_pred = model.predict(x_val)
        mse = mean_squared_error(y_val,y_pred)
        loss = model.evaluate(x_val, y_val)
        save_model(model)
        with open("server_results.csv", 'a') as fw:
            fw.write(f"{mse}, {loss}\n")
        return loss, {"mse":mse}

    return evaluate

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    mse_scores = [num_examples * m["mse"] for num_examples, m in metrics]
    # recalls = [num_examples * m["recall"] for num_examples, m in metrics]
    # f1_scores = [num_examples * m["f1_score"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {
        "mse": sum(mse_scores) / sum(examples)
    }

def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one local epoch,
    increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 32,
        "local_epochs": 1 if server_round < 2 else 2,
    }
    return config


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five batches) during
    rounds one to three, then increase to ten local evaluation steps.
    """
    val_steps = 5 if server_round < 4 else 10
    return {"val_steps": val_steps}


if __name__ == "__main__":
    main()