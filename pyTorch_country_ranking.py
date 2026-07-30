import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load data
csv_path = "./data/global_country_rankings_2000_2026-selected-columns.csv"
df = pd.read_csv(csv_path)
print("Columns:", list(df.columns))

# Features
cat_features = ["Country", "Region"]
num_features = [
    "Year",
    "Economic_Tier",
    "Global_Hunger_Rank",
    "GDP_Per_Capita_Rank",
    "Life_Expectancy_Rank",
    "Corruption_Perception_Rank",
]
y_feature = "Happiness_Rank"

X = df[cat_features + num_features]
y = df[y_feature]
print(f"X shape: {X.shape} | y shape: {y.shape}")

# Split data
RANDOM_SEED = 42
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.05,
    random_state=RANDOM_SEED,
)

# Preprocessing pipeline
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", Ridge(alpha=1.0, random_state=RANDOM_SEED)),
    ]
)

# Train
model.fit(X_train, y_train)

# Evaluate
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

train_mse = mean_squared_error(y_train, train_preds)
test_mse = mean_squared_error(y_test, test_preds)
train_r2 = r2_score(y_train, train_preds)
test_r2 = r2_score(y_test, test_preds)

print("Training set")
print(f"  MSE: {train_mse:.4f}")
print(f"  R2 : {train_r2:.4f}")
print("Test set")
print(f"  MSE: {test_mse:.4f}")
print(f"  R2 : {test_r2:.4f}")
