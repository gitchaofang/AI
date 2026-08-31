import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# down load data
df = pd.read_csv("./data/global_country_rankings_2000_2026-selected-columns.csv")
print(df.columns)
cat_features = ["Country","Region"]
num_features = ["Year", "Economic_Tier","Global_Hunger_Rank","GDP_Per_Capita_Rank","Life_Expectancy_Rank","Corruption_Perception_Rank"]
y_feature = ["Happiness_Rank"]
X_num = df[num_features]
X_cat = df[cat_features]
y = df[y_feature]

# preprocessing
num_imputer = SimpleImputer(strategy="mean")
cat_imputer = SimpleImputer(strategy="most_frequent")
encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore")
scaler = StandardScaler()
X_num = scaler.fit_transform(num_imputer.fit_transform(X_num))
X_cat = encoder.fit_transform(cat_imputer.fit_transform(X_cat))
X = np.hstack([X_num,X_cat])
X_sample = torch.tensor(X, dtype = torch.float32)
y_sample = torch.tensor(y.to_numpy(), dtype = torch.float32)
print(f"X shape: {X.shape} | y shape: {y.shape}")

# Hyperparameters
INPUT_FEATURERS = len(X[0])
HIDDEN_DIM =10
RANDOM_SEED = 42
# create DataLoader
X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.05, random_state=RANDOM_SEED)
dataset_train = TensorDataset(X_train,y_train)
dataset_test = TensorDataset(X_test,y_test)

dataloader_train = DataLoader(
    dataset_train,
    batch_size = 32,
    shuffle = True,
)
dataloader_test = DataLoader(
    dataset_test,
    batch_size = 32,
    shuffle = True,
)
print(f"size of train dataset is: {len(dataloader_train)} | size of test dataset is: {len(dataset_test)}")

#model
class LinearRegNNModel(nn.Module):
    def __init__(self, in_features = INPUT_FEATURERS, out_features = 1, hidden_dim =HIDDEN_DIM):
        super().__init__()
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features = INPUT_FEATURERS, out_features = HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(in_features = HIDDEN_DIM, out_features = HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(in_features = HIDDEN_DIM, out_features = HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(in_features = HIDDEN_DIM, out_features = 1),
        )
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        return self.linear_layers(x)

model = LinearRegNNModel()

# setup for training
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(params = model.parameters(), lr = 0.001)

# training loop
epochs = 500
for epoch in range(epochs):
    train_loss_acc = 0.0
    test_loss_acc = 0.0
    for X,y in (dataloader_train):
        model.train()
        # logits
        logits = model(X)
        # loss
        loss = loss_fn(logits, y)
        train_loss_acc += loss.item()
        # zero grad
        optimizer.zero_grad()
        # backpropagation
        loss.backward()
        # update
        optimizer.step()
    model.eval()
    with torch.inference_mode():
        for X_test, y_test in (dataloader_test):
            # logits
            logits = model(X_test)
            # loss
            loss_test = loss_fn(logits,y_test)
            test_loss_acc += loss_test
    print(f"epoch {epoch} | training loss: {train_loss_acc / len(dataloader_train)} | test loss: {test_loss_acc / len(dataloader_test)}")