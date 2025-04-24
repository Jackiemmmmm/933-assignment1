import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from log import log_output
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt


log_output()
# Load dataset
df = pd.read_csv("bike_rental_data.csv")

df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)

df["workingday_hour_sin"] = df["workingday"] * df["sin_hour"]
df["workingday_hour_cos"] = df["workingday"] * df["cos_hour"]

df["temp_squared"] = df["temp"] ** 2

# Features and target
X = df.drop(columns=["bikes_rented"])
y = df["bikes_rented"]

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df["season"]):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]


X_train = strat_train_set.drop(columns=["bikes_rented"])
y_train = strat_train_set["bikes_rented"]
X_test = strat_test_set.drop(columns=["bikes_rented"])
y_test = strat_test_set["bikes_rented"]


categorical_features = ["workingday", "holiday"]
numerical_features = [
    "temp",
    "humidity",
    "windspeed",
    "hour",
    "sin_hour",
    "cos_hour",
    "temp_squared",
]


preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(), categorical_features),
    ]
)

# Standardize features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)


X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# --- Statistical Learning ---
# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train_processed, y_train)


# first test alpha list - best 0.1
# param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
# second test alpha list - best 0.3
# param_grid = {"alpha": [0.1, 0.2, 0.3, 0.4, 0.5]}
# third test alpha list - best 0.31
param_grid = {"alpha": [0.31, 0.32, 0.33, 0.34, 0.35]}
ridge_cv = GridSearchCV(Ridge(), param_grid, cv=5, scoring="neg_mean_squared_error")
ridge_cv.fit(X_train_processed, y_train)
best_alpha = ridge_cv.best_params_["alpha"]


# Ridge Regression
ridge = Ridge(alpha=0.31)
ridge.fit(X_train_processed, y_train)

# Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_processed, y_train)


# Elastic Net
# elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
# elastic_net.fit(X_train_processed, y_train)


# use GridSearchCV to find the best alpha and l1_ratio
# first test alpha list - best alpha 0.001, l1_ratio 0.9
# param_grid = {
#     "alpha": [0.001, 0.005, 0.01, 0.015, 0.02],
#     "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
# }
# second test alpha list - best alpha 0.03, l1_ratio 0.96
# param_grid = {
#     "alpha": [0.02, 0.03, 0.04, 0.05, 0.06],
#     "l1_ratio": [0.92, 0.93, 0.94, 0.95, 0.96],
# }
# third test alpha list - best alpha 0.03, l1_ratio 1.0
param_grid = {
    "alpha": [0.029, 0.03, 0.031, 0.032, 0.033],
    "l1_ratio": [0.96, 0.97, 0.98, 0.99, 1.0],
}


elastic_net_cv = GridSearchCV(
    ElasticNet(max_iter=10000), param_grid, cv=5, scoring="neg_mean_squared_error"
)
elastic_net_cv.fit(X_train_processed, y_train)

# Evaluate the best model
best_alpha = elastic_net_cv.best_params_["alpha"]
best_l1_ratio = elastic_net_cv.best_params_["l1_ratio"]


# Train the best Elastic Net model
elastic_net = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio)
elastic_net.fit(X_train_processed, y_train)

# Evaluate models
def evaluate(model, X, y, name):
    y_pred = model.predict(X)
    print(
        f"{name}: RMSE={root_mean_squared_error(y, y_pred):.2f}, R^2={r2_score(y, y_pred):.2f}"
    )


evaluate(lin_reg, X_test_processed, y_test, "Linear Regression")
print("")
print(f"Best Ridge Regression alpha (λ): {best_alpha}")
print("")
evaluate(ridge, X_test_processed, y_test, "Ridge Regression")
evaluate(lasso, X_test_processed, y_test, "Lasso Regression")

feature_names = numerical_features + [
    f"{cat}_{val}" for cat in categorical_features for val in [0, 1]
]
evaluate(elastic_net, X_test_processed, y_test, "Elastic Net")

print('')
print(f"best alpha: {best_alpha}, best l1_ratio: {best_l1_ratio}")
print('')

# Evaluate the best Elastic Net model
coef_comparison = pd.DataFrame(
    {"Linear": lin_reg.coef_,"Ridge": ridge.coef_, "Lasso": lasso.coef_, "ElasticNet": elastic_net.coef_},
    index=feature_names,
)

print(coef_comparison)
print('')



# --- Deep Learning Approach ---
class LinearNN(nn.Module):
    def __init__(self, input_dim):
        super(LinearNN, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # Single-layer linear model

    def forward(self, x):
        return self.linear(x)


# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_processed, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Initialize model, loss, and optimizer
model = LinearNN(X_train_processed.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(
    model.parameters(), lr=0.01, weight_decay=0.01
)  # L2 regularization (weight decay)

# Train model
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {np.sqrt(loss.item()):.4f}")

# Evaluate on test data
model.eval()
y_pred_tensor = model(X_test_tensor).detach().numpy()
test_rmse = root_mean_squared_error(y_test, y_pred_tensor)
test_r2 = r2_score(y_test, y_pred_tensor)
print(f"Deep Learning Model: RMSE={test_rmse:.2f}, R^2={test_r2:.2f}")
