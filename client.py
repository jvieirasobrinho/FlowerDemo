import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict

# Simple model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create model and some dummy data
model = SimpleNet()
X_train = torch.randn(100, 10)
y_train = torch.randint(0, 2, (100,))
trainloader = DataLoader(TensorDataset(X_train, y_train), batch_size=32)

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in model.state_dict().items()]
    
    def set_parameters(self, parameters):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        # Simple training loop
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        model.train()
        
        for _ in range(1):  # 1 epoch
            for batch in trainloader:
                X, y = batch
                optimizer.zero_grad()
                loss = F.cross_entropy(model(X), y)
                loss.backward()
                optimizer.step()
        
        return self.get_parameters(config={}), len(trainloader.dataset), {}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        model.eval()
        
        loss = 0
        with torch.no_grad():
            for X, y in trainloader:
                loss += F.cross_entropy(model(X), y).item()
        
        return loss / len(trainloader), len(trainloader.dataset), {"loss": loss}

# Connect to server
# Replace with your DGX server IP address from Step 4 of server setup
fl.client.start_client(
    server_address="169.226.129.103:8888",
    client=FlowerClient().to_client()
)