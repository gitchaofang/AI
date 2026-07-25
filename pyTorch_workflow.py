import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# check torch version
print(torch.__version__)

# setup
start = 0
end = 20
step = 0.2

weight = 0.7
bias = 0.3

X = torch.arange(start, end, step).unsqueeze(axis = 1)
Y = weight * X + bias

print(f"X[:10]: {X[:10]} ")
print(f"Y[:10]: {Y[:10]} ")
print(f"len(X): {len(X)} ")
print(f"len(Y): {len(Y)} ")

# spliting data into training and testing sets
train_split = int(0.8 * len(X))
train_x, train_y = X[:train_split], Y[:train_split]
test_x, test_y = X[train_split:], Y[train_split:]
print(f"len(train_x): {len(train_x)} ")
print(f"len(test_x): {len(test_x)} ")
print(f"len(train_y): {len(train_y)} ")
print(f"len(test_y): {len(test_y)} ")

#build a linear regression model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # paramaters 
        self.weights = nn.Parameter(torch.randn(size = (1,),
                                                 requires_grad = True,
                                                 dtype = torch.float32))
        self.bias = nn.Parameter(torch.randn(size = (1,),
                                               requires_grad = True,
                                               dtype = torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias


# Create model directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True,exist_ok=True)

#Create model save path
MODEL_NAME = "model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH/MODEL_NAME

save = False
load = True

# checking the model
torch.manual_seed(42)
model_0 = LinearRegressionModel()
if load:
    model_0.load_state_dict(torch.load(MODEL_SAVE_PATH))
    print(f"Loaded model from {MODEL_SAVE_PATH} ")
print(f"model weights are: {model_0.state_dict()['weights']} ")
print(f"model bias are: {model_0.state_dict()['bias']} ")

# prediction 
with torch.inference_mode():
    y_preds = model_0(test_x)
print(f"predictions: {y_preds[:10]} ")

# define loss function
loss_fn = nn.MSELoss()
#define optimizer
optimizer = torch.optim.SGD(params = model_0.parameters(),
                              lr = 0.001)

epochs = 1
test_loss = []
train_loss = []

for epoch in range(epochs):

    model_0.train()
    #frowarrd pass
    y_preds = model_0(train_x)
    #loss
    loss = loss_fn(y_preds, train_y)
    #optimizer step
    optimizer.zero_grad()
    #backpropagation
    loss.backward()
    #update
    optimizer.step()

   # print(f"loss of epoch {epoch} is: {loss}")

    # eval
    model_0.eval()
    with torch.inference_mode():
        if(epoch % 10 == 0):
            test_y_prerds = model_0(test_x)
            test_loss = loss_fn(test_y_prerds,test_y)
            print(f"Epoch: {epoch} | training loss: {loss} | testing loss: {test_loss}")
            print(f"{model_0.state_dict()}")

        #
print(f"model weights are: {model_0.state_dict()}")
print(f"real parameters are: {0.7,0.3}")
if save:
    torch.save(obj = model_0.state_dict(),
               f=MODEL_SAVE_PATH)
