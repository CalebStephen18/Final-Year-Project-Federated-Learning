# Federated Learning Temperature Prediction System

## Overview

This project implements a federated learning system for temperature prediction using TensorFlow and Flower (FL). Federated learning is a machine learning approach where the model is trained across multiple decentralized devices (clients) holding local data samples, without exchanging them with a central server.

### Components

- `client.py`: Defines the client-side logic, including data loading, model training, and evaluation.
- `server.py`: Implements the federated learning server, coordinating model aggregation and evaluation.
- `webapp.py`: Offers a Streamlit-based web application for predicting comfortable temperatures.
- `client_even.py` and `client_odd.py`: Alternate client scripts for running federated learning with even and odd client IDs.
- `datasetz/`: Directory containing temperature datasets for different cities.

## Setup

1. **Installation**: Install the required Python packages pandas, flwr, tensorflow, keras, numpy, streamlit.
2. **Training**: Start the server by running `python server.py` and then run the desired clients (`client.py`, `client_even.py`, or `client_odd.py`).
3. **Web Application**: Run `webapp.py` and access the application through the provided URL.

## Usage

- **Client**: No additional configuration is required for the client. It automatically loads the dataset, trains the model, and sends updates to the server.
- **Server**: The server orchestrates the federated learning process. It aggregates model updates from clients and evaluates the global model's performance.
- **Web Application**: Users can input location, date, and time to get a comfortable temperature prediction based on federated learning models.

## Contributors

- Main Author: Caleb Stephen
- Co-Author: Richie Suresh Koshy


