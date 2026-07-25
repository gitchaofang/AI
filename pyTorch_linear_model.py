import torch 
from torch import nn
from pathlib import Path

torch.manual_seed(42)
# create data set
SAMPLE_SIZE = 100
FEATURE_NUM = 4
X = torch.randn(size = (SAMPLE_SIZE,FEATURE_NUM),dtype = torch.float32)
weight = torch.randn(size = (FEATURE_NUM,),dtype = torch.float32)
bias = torch.randn(size = (1,),dtype = torch.float32)
Y = torch.matmul(X,weight).unsqueeze(-1)
print(f"Y shape: {Y.shape}")
train_split = int(SAMPLE_SIZE * 0.7)
train_x, train_y = X[:train_split,:], Y[:train_split,:]
test_x, test_y = X[train_split:,:], Y[train_split:,:]
print(f"train size: {len(train_y)}")
print(f"test size: {len(test_y)}")
print(f"train_x shape: {train_x.shape}")
#print(f"{train_y[:30]}")

# Creat model direrctory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents = True, exist_ok = True)

# Create model save path
MODEL_NAME = "linear_model.pth"
MODEL_SAVE_PATH = MODEL_PATH/MODEL_NAME

#Setups
save = False
load = False
input_dim = FEATURE_NUM
output_dim = 1

# Create linear model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features = input_dim, out_features = output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)

model = LinearRegressionModel()
if load:
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    print(f"load model from: {MODEL_SAVE_PATH}")

# Build training loop
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(params = model.parameters(), lr = 0.001)
epochs = 10000
for epoch in range(epochs):
    model.train()
    #forward pass
    y_preds = model(train_x)
    # loss
    loss = loss_fn(y_preds,train_y)
    #optimizer zero grad
    optimizer.zero_grad()
    #backpropagation
    loss.backward()
    # param update
    optimizer.step()

    #evaluate
    model.eval()
    with torch.inference_mode():
        if(epoch % 10 == 0):
            eval_y = model(test_x)
            eval_loss = loss_fn(eval_y, test_y)
            print(f"Epoch {epoch} | training_loss: | {loss} | testing loss: {eval_loss}")
print(f"original weights: {weight} | orriginal bias: {bias}")
print(f"trained model: {model.state_dict()}")
if save:
    torch.save(obj = model.state_dict(),f = MODEL_SAVE_PATH)
    print(f"Linear model is saved")
