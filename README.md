# FlowerDemo

Please make sure to change NetIDs, IP addresses, port numbers, lab directories, and any other information as needed.

## Part 1: Server Setup (on DGX)

### Step 1: Connect to DGX and Start Interactive Session

```bash
ssh jv535825@dgx-head01.its.albany.edu
srun --time=08:00:00 --gpus=1 --pty /bin/bash
```

### Step 2: Create and Activate Conda Environment

```bash
conda create --name flower
conda activate flower
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install flwr
```

Verify installation:
```bash
python -c "import flwr;print(flwr.__version__)"
```

### Step 3: Create Server Directory and Script

Navigate to your lab directory:
```bash
cd /network/rit/dgx/dgx_vieirasobrinho_lab
mkdir flower
cd flower
```

Create the server script:
```bash
vim server.py
```

Copy and paste this code:
```python
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
```

Save and exit (`:wq` in vim).

### Step 4: Get Server IP Address

Before starting the server, get the IP address that clients will connect to:
```bash
ip -br a | grep enp225s0f0np0
```

**Save this IP address** - you'll need it for the client setup. Example: `169.226.129.103`

### Step 5: Start the Server

```bash
python server.py
```

You should see: `Server running on port 8888`

**Keep this terminal open** - the server needs to stay running while clients connect.

---

## Part 2: Client Setup (on Your Local Computer)

### Step 1: Create and Activate Conda Environment

Open a new terminal on your local machine:
```bash
conda create --name flower
conda activate flower
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install flwr pytorch
```

Verify installation:
```bash
python -c "import flwr;print(flwr.__version__)"
```

### Step 2: Create Client Directory and Script

```bash
mkdir ~/flower
cd ~/flower
vim client.py
```

Copy and paste this code:
```python
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
```

**Important:** Replace `169.226.129.103` with the IP address you got from the server setup.

Save and exit.

### Step 3: Start the Client

```bash
python client.py
```

The client will connect to the server and participate in federated learning!

---

## What Happens Next?

1. The server sends the initial model to the client
2. The client trains on its local data for 1 epoch
3. The client sends updated model weights back to the server
4. This repeats for 3 rounds (configured in `num_rounds=3`)
5. Both programs will exit when training is complete
