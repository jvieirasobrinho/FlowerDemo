import flwr as fl

# Define strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=1.0,  # Sample 100% for evaluation
    min_fit_clients=1,  # Minimum number of clients for training
    min_evaluate_clients=1,  # Minimum number for evaluation
    min_available_clients=1,  # Wait for at least 1 client
)

# Start Flower server
if __name__ == "__main__":
    fl.server.start_server(
        server_address="0.0.0.0:8888",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )
    print("Server running on port 8888")