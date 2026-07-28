import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

# get datasets
train_data = datasets.FashionMNIST(root = "data",
                                  train = True,
                                  download = True,
                                  transform = ToTensor(),
                                  target_transform = None)

test_data = datasets.FashionMNIST(root = "data",
                                  train = False,
                                  download = True,
                                  transform = ToTensor())
class_names = train_data.classes

image, label = train_data[0]

# set up dataloader
BATCH_SIZE = 32

train_dataloader = DataLoader(train_data,
                              batch_size = BATCH_SIZE,
                              shuffle = True)
test_dataloader = DataLoader(test_data,
                             batch_size = BATCH_SIZE,
                             shuffle = False)


# build cnn model
INPUT_FEATURES = 1
OUTPUT_FEATURES = len(class_names)
class CNNModel(nn.Module):
    def __init__(self, in_features: int, hidden_units: int, out_features: int):
        super().__init__()
        self.cnn_block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=hidden_units, kernel_size=(3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=(3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride = 2),
        )
        self.cnn_block2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=(3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=(3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride = 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7, out_features=OUTPUT_FEATURES),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn_block2(self.cnn_block1(x))
        x = self.classifier(x)
        return x

model = CNNModel(in_features=INPUT_FEATURES, hidden_units=10, out_features=OUTPUT_FEATURES)

# setups
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params = model.parameters(), lr = 0.001)
epochs = 50
for epoch in range(epochs):
    train_loss = 0.0
    for batch_idx, (X, y) in enumerate(train_dataloader):
        model.train()
        # logits
        y_logits = model(X)
        # loss
        loss = loss_fn(y_logits, y)
        train_loss += loss.item()
        # zero grad
        optimizer.zero_grad()
        # backpropagation
        loss.backward()
        # update
        optimizer.step()
    train_loss /= len(train_dataloader)

    # test 
    loss_test_acc = 0.0
    model.eval()
    with torch.inference_mode():
        for X,y in(test_dataloader):
            # test logitsß
            logits_test = model(X)
            # test loss
            loss_test = loss_fn(logits_test, y)
            loss_test_acc+=loss_test.item()
    loss_test_acc /= len(test_dataloader)
    print(f"epoch {epoch} | training loss: {train_loss} | test loss: {loss_test_acc}")
        
