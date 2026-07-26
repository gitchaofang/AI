import torch
from torch import nn
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# Hyperparameters 
NUM_CLASSES = 8
NUM_FEATURES = 4
RANRDOM_SEED = 42

# Create multi-class data
X_blob, y_blob = make_blobs(n_samples=1000,
                            n_features=NUM_FEATURES,
                            centers = NUM_CLASSES,
                            cluster_std=1.5,
                            random_state=RANRDOM_SEED)
# Turn data into tensors
X_blob = torch.from_numpy(X_blob).type(torch.float32)
y_blob = torch.from_numpy(y_blob).type(torch.int64)
#print(f"X_blob and type: {X_blob[:10,:]},{X_blob.dtype}")
#print(f"y_blob and type: {y_blob[:10,:]},{y_blob.dtype}")
X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob, y_blob,test_size = 0.2, random_state=RANRDOM_SEED)
#print(f"x_train size: {len(X_blob_train)}. x_train shape: {X_blob_train.shape}")
#print(f"y_train size: {len(y_blob_train)}. y_train shape: {y_blob_train.shape}")

# Build model
class MultiClassificationModel(nn.Module):
    def __init__(self, in_features, out_features, hidden_unites = 8):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features,hidden_unites),
            nn.ReLU(),
            nn.Linear(hidden_unites,hidden_unites),
            nn.ReLU(),
            nn.Linear(hidden_unites,out_features),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer_stack(x)

model = MultiClassificationModel(NUM_FEATURES,NUM_CLASSES)

def accuracy_fn(y_pred,y_true):
    correct = torch.eq(y_pred,y_true).type(torch.float32).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

# setups:
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr = 0.001)
print(f"y_blob_train type: {y_blob_test.shape}")
# training loop:
epochs = 10000
for epoch in range(epochs):
    model.train()
    # logits
    y_logits = model(X_blob_train)
    # y_pred
    y_pred = torch.argmax(torch.softmax(y_logits, axis = 1), axis = 1)
    # loss
    loss = loss_fn(y_logits, y_blob_train)
    # accuracy
    acc = accuracy_fn(y_pred, y_blob_train)
    # optimizer
    optimizer.zero_grad()
    # loss backpropagation
    loss.backward()
    # update
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        # forwarrd pass
        test_logits = model(X_blob_test)
        test_pred = torch.argmax(torch.softmax(test_logits, axis = 1),axis = 1)
        test_acc = accuracy_fn(test_pred, y_blob_test)
        test_loss = loss_fn(test_logits,y_blob_test)
        if epoch % 10:
            print(f"Epoch: {epoch} |Train Loss {loss}| Train accuracy is: {acc} % | Test loss: {test_loss} | accuracy is: {test_acc} %")
    

