import torch
from torch import nn
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

#Hyperparameters
NUM_CLASSES = 4
NUM_FEATURES = 4
NUM_SAMPLES = 10000
RANDOM_SEED = 42

#make data
X_blob, y_blob = make_blobs(n_samples = NUM_SAMPLES,
                           n_features = NUM_FEATURES,
                           centers = NUM_CLASSES,
                           cluster_std = 1.5,
                           random_state = RANDOM_SEED)
X_blob = torch.tensor(X_blob, dtype = torch.float32)
y_blob = torch.tensor(y_blob, dtype = torch.int64)

#split
X_train, X_test, y_train, y_test = train_test_split(X_blob, y_blob, test_size = 0.2, random_state= RANDOM_SEED)
print(f"X_train shape: {X_train.shape}")

#build model
class MultiClassificationModel(nn.Module):
    def __init__(self, in_features, n_classes, hidden_unites = 10):
        super().__init__()
        self.nn_layer = nn.Sequential(
            nn.Linear(in_features = in_features, out_features = hidden_unites),
            nn.ReLU(),
            nn.Linear(in_features = hidden_unites, out_features = hidden_unites),
            nn.ReLU(),
            nn.Linear(in_features = hidden_unites, out_features = n_classes),
        )
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.nn_layer(x)
model = MultiClassificationModel(NUM_FEATURES,NUM_CLASSES)

def accuracy_fn(y_pred,y_true):
    correct = torch.eq(y_pred,y_true).sum().item()
    acc = (correct / len (y_pred)) * 100
    return acc

# setups for training
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr = 0.001)

#training loop
epochs = 10000
for epoch in range(epochs):
    model.train()
    # forwarrd
    logits = model(X_train)
    # loss
    loss = loss_fn(logits,y_train)
    # y_pred
    y_pred = torch.argmax(torch.softmax(logits,axis = 1), axis = 1)
    # accuracy
    acc = accuracy_fn(y_pred, y_train)
    # zero grad
    optimizer.zero_grad()
    # backpropagation
    loss.backward()
    # update
    optimizer.step()

    if epoch % 40 == 0:
        model.eval()
        with torch.inference_mode():
            # forward
            logits_test = model(X_test)
            # loss
            loss_test = loss_fn(logits_test, y_test)
            # pred
            pred_test = torch.argmax(torch.softmax(logits_test, axis = 1), axis = 1)
            # accuracy
            acc_test = accuracy_fn(pred_test,y_test)
            print(f"Epoch: {epoch} |Train Loss {loss}| Train accuracy is: {acc} % | Test loss: {loss_test} | accuracy is: {acc_test} %")
    

