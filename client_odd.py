import flwr as fl
from client import FlowerClient

if __name__ == "__main__":
    DATASET_PATH = "./datasetz"
    client_id = 1
    client = FlowerClient(client_id=client_id, dataset_path=DATASET_PATH)

    fl.client.start_numpy_client(server_address="127.0.0.1:8087", client=client)

    print(f"Client {client_id} training completed.")

    print("All clients have completed training.")

