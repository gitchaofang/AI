import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# read data
df = pd.read_csv("./data/fifa_world_cup_2026_player_performance.csv")
columns = df.columns
feature_cols = ["age","nationality", "height_cm","weight_kg","preferred_foot","total_goals_tournament"]
#feature_cols = ["age","nationality","height_cm","total_goals_tournament"]
pred_cols = ["tournament_rating"]
X = df[feature_cols]
y = df[pred_cols]
#print(f"X: {X.head} | y: {y.head()}")

# preprocessing
# numerical cols
num_cols = ["age","height_cm","weight_kg","total_goals_tournament"]
#num_cols = ["age","height_cm","total_goals_tournament"]
X_num = X[num_cols]
num_imputer = SimpleImputer(strategy="mean")
scaler = StandardScaler()
X_num = num_imputer.fit_transform(X_num)
X_num = scaler.fit_transform(X_num)

# card cols
card_cols = ["nationality","preferred_foot"]
#card_cols = ["nationality"]
X_card = X[card_cols]
card_imputer =SimpleImputer(strategy = "most_frequent")
X_card = card_imputer.fit_transform(X_card)
encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore"
)
X_card = encoder.fit_transform(X_card)
X = np.hstack([X_num,X_card])


# turn into pytorch
X = torch.tensor(X, dtype = torch.float32)
y = torch.tensor(y.to_numpy(), dtype = torch.float32)

# hyperparameters
INPUT_FEATURES = len(X[0])
HIDDEN_DIM = 10
RANDOM_SEED = 42

# build a dataloader
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2,random_state=RANDOM_SEED)
train_dataset = TensorDataset(X_train,y_train)
train_loader = DataLoader(
    train_dataset,
    batch_size = 1024,
    shuffle = True,
)

test_dataset = TensorDataset(X_test,y_test)
test_loader = DataLoader(
    test_dataset,
    batch_size = 1024,
    shuffle = False,
)

# model
class LinearRegModel(nn.Module):
    def __init__(self, in_features = INPUT_FEATURES, out_features = 1, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features = in_features, out_features = hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features = hidden_dim, out_features = hidden_dim),
            nn.ReLU(),
             nn.Linear(in_features = hidden_dim, out_features = hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features = hidden_dim, out_features = 1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layers(x)

model = LinearRegModel(INPUT_FEATURES, 1, HIDDEN_DIM)

# setup for training
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(params = model.parameters(), lr = 0.0005)

# taining loop
epochs = 100
for epoch in range(epochs):
    train_loss_acc = 0.0
    test_loss_acc = 0.0
    for X,y in (train_loader):
        model.train()
        # output
        out = model(X)
        # loss
        loss = loss_fn(out,y)
        train_loss_acc += loss.item()
        # zero grad
        optimizer.zero_grad()
        # backgropagation
        loss.backward()
        # update
        optimizer.step()
    for X_test, y_test in (test_loader):
        # output
        out_test = model(X_test)
        # loss
        loss_test = loss_fn(out_test,y_test)
        test_loss_acc += loss_test.item() 
    train_loss_acc /= len(train_dataset)
    test_loss_acc /= len(test_dataset)
    print(f"epoch {epoch} | training loss: {train_loss_acc} | test loss: {test_loss_acc}")