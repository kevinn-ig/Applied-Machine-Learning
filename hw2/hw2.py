import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

abalone_data = pd.read_csv('/Users/kevin_smith/Desktop/FSU_Relevant_Stuff/spring23/STA5635/homework/hw2/abalone.csv')

# Split data into predictors and response variables
X = abalone_data.iloc[:, :7]
y = abalone_data.iloc[:, -1]

# Number of random splits
num_splits = 20

## Code for Question 1
# Create a list to store the MSE values for training and testing sets
train_mse_list = []
test_mse_list = []

for _ in range(num_splits):
  # Random split 
  X_train, X_test, y_train, y_test = train_test_split(X, 
                      y,
                      test_size = 0.1, 
                      random_state = np.random.randint(1,100))

  # Calculate average training y
  avg_train_y = np.mean(y_train)

  # Predict test set responses using average training y
  y_pred_train = np.full_like(y_train, avg_train_y)
  y_pred_test = np.full_like(y_test, avg_train_y)

  # Calculate MSE
  train_mse = mean_squared_error(y_train, y_pred_train)
  test_mse = mean_squared_error(y_test, y_pred_test)

  # Append MSE values to lists
  train_mse_list.append(train_mse)
  test_mse_list.append(test_mse)


# Calculate average MSE for training and testing sets across the 20 splits
avg_null_train_mse = np.mean(train_mse_list)
avg_null_test_mse = np.mean(test_mse_list)

print(f"Average training MSE for Null Model: {avg_null_train_mse}")
print(f"Average testing MSE for Null Model: {avg_null_test_mse}")


## Code for Question 2
# Create a list to store metrics
train_r2_list, test_r2_list = [], []
train_mse_list, test_mse_list = [], []
log_det_list = []

for _ in range(num_splits):
  # Random split 
  X_train, X_test, y_train, y_test = train_test_split(X,
                      y,
                      test_size = 0.1,
                      random_state = np.random.randint(1,100))

  # Perform Ridge Regression
  lambda_value = 0.001
  XTX_plus_lambdaIp = np.dot(X_train.T, X_train) + lambda_value * np.identity(X_train.shape[1])

  ridge_weights = np.linalg.solve(XTX_plus_lambdaIp, np.dot(X_train.T, y_train))

  # Model Evaluation
  y_pred_train = np.dot(X_train, ridge_weights)
  y_pred_test = np.dot(X_test, ridge_weights)

  train_r2 = r2_score(y_train, y_pred_train)
  test_r2 = r2_score(y_test, y_pred_test)

  train_mse = mean_squared_error(y_train, y_pred_train)
  test_mse = mean_squared_error(y_test, y_pred_test)

  # Calculate Log Determinant 
  log_det = np.log(np.linalg.det(XTX_plus_lambdaIp))

  # Append metrics to lists
  train_r2_list.append(train_r2)
  test_r2_list.append(test_r2)
  train_mse_list.append(train_mse)
  test_mse_list.append(test_mse)
  log_det_list.append(log_det)


# Calculate average and standard deviation for metrics
avg_train_r2 = np.mean(train_r2_list)
std_train_r2 = np.std(train_r2_list)

avg_test_r2 = np.mean(test_r2_list)
std_test_r2 = np.std(test_r2_list)

avg_train_mse = np.mean(train_mse_list)
std_train_mse = np.std(train_mse_list)

avg_test_mse = np.mean(test_mse_list)
std_test_mse = np.std(test_mse_list)

avg_log_det = np.mean(log_det_list)
std_log_det = np.std(log_det_list)

# Print results
print(f"Average Training R^2: {avg_train_r2}, Std Training R^2: {std_train_r2}")
print(f"Average Testing R^2: {avg_test_r2}, Std Testing R^2: {std_test_r2}")
print(f"Average Training MSE: {avg_train_mse}, Std Training MSE: {std_train_mse}")
print(f"Average Testing MSE: {avg_test_mse}, Std Testing MSE: {std_test_mse}")
print(f"Average Log Determinant: {avg_log_det}, Std Log Determinant: {std_log_det}")


## Code for Question 3
max_depths = range(1,8)

# List to store metrics
train_r2_avg, test_r2_avg = [], []
train_mse_avg, test_mse_avg = [], []

null_model_mse = avg_null_test_mse

for depth in max_depths:
  train_r2_list, test_r2_list = [], []
  train_mse_list, test_mse_list = [], []

  for _ in range(num_splits):
    # Random Split
    X_train, X_test, y_train, y_test = train_test_split(X,
                      y,
                      test_size=0.1,
                      random_state=np.random.randint(1,100))

    # Decision Tree Regression
    dt_model = DecisionTreeRegressor(max_depth=depth)
    dt_model.fit(X_train, y_train)

    # Model Evaluation
    y_pred_train = dt_model.predict(X_train)
    y_pred_test = dt_model.predict(X_test)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)

    # Append metrics
    train_r2_list.append(train_r2)
    test_r2_list.append(test_r2)
    train_mse_list.append(train_mse)
    test_mse_list.append(test_mse)

  # Calculate average metric for current tree depth
  avg_train_r2 = np.mean(train_r2_list)
  avg_test_r2 = np.mean(test_r2_list)
  avg_train_mse = np.mean(train_mse_list)
  avg_test_mse = np.mean(test_mse_list)

  # Append average metrics to lists
  train_r2_avg.append(avg_train_r2)
  test_r2_avg.append(avg_test_r2)
  train_mse_avg.append(avg_train_mse)
  test_mse_avg.append(avg_test_mse)

# Plot R^2 vs Tree Depth
plt.figure(figsize= (10,6))
plt.plot(max_depths, train_r2_avg, label="Training R^2")
plt.plot(max_depths, test_r2_avg, label="Testing R^2")
plt.xlabel("Tree Depth")
plt.ylabel("R^2")
plt.title("Average Training and Testing R^2 vs Tree Depth")
plt.legend()
plt.savefig("r2_vs_depth.png")

# Plot MSE vs Tree Depth with Null Model MSE as a horizontal line
plt.figure(figsize= (10,6))
plt.plot(max_depths, train_mse_avg, label="Training MSE")
plt.plot(max_depths, test_mse_avg, label="Testing MSE")
plt.axhline(y=null_model_mse, color='r', linestyle='--', label="Null Model MSE")
plt.xlabel("Tree Depth")
plt.ylabel("MSE")
plt.title("Average Training and Testing MSE vs Tree Depth")
plt.legend()
plt.savefig("MSE_vs_depth.png")


## Code for Question 4
num_trees_list = [10, 30, 100, 300]

# Lists to store results
results = {}

for num_trees in num_trees_list:
  train_r2_list, test_r2_list = [], []
  train_mse_list, test_mse_list = [], []

  for _ in range(num_splits):
    # Random Split
    X_train, X_test, y_train, y_test = train_test_split(X,
                    y,
                    test_size=0.1,
                    random_state = np.random.randint(1, 100))

    # Random Forest Regression
    rf_model = RandomForestRegressor(n_estimators=num_trees)
    rf_model.fit(X_train, y_train)

    # Model Evaluation
    y_pred_train = rf_model.predict(X_train)
    y_pred_test = rf_model.predict(X_test)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)

    # Append metrics
    train_r2_list.append(train_r2)
    test_r2_list.append(test_r2)
    train_mse_list.append(train_mse)
    test_mse_list.append(test_mse)

  # Calculate average and std for each metric
  avg_train_r2 = np.mean(train_r2_list)
  std_train_r2 = np.std(train_r2_list)

  avg_test_r2 = np.mean(test_r2_list)
  std_test_r2 = np.std(test_r2_list)

  avg_train_mse = np.mean(train_mse_list)
  std_train_mse = np.std(train_mse_list)

  avg_test_mse = np.mean(test_mse_list)
  std_test_mse = np.std(test_mse_list)

  # Store results in dictionary
  results[num_trees] = {
    'avg_train_r2': avg_train_r2,
    'std_train_r2': std_train_r2,
    'avg_test_r2': avg_test_r2,
    'std_test_r2': std_test_r2,
    'avg_train_mse': avg_train_mse,
    'std_train_mse': std_train_mse,
    'avg_test_mse': avg_test_mse,
    'std_test_mse': std_test_mse
  }

# Print results
for num_trees, metrics in results.items():
  print(f"Number of Trees: {num_trees}")
  print(f"Average Training R^2: {metrics['avg_train_r2']}, Std Training R^2: {metrics['std_train_r2']}")
  print(f"Average Testing R^2: {metrics['avg_test_r2']}, Std Testing R^2: {metrics['std_test_r2']}")
  print(f"Average Training MSE: {metrics['avg_train_mse']}, Std Training MSE: {metrics['std_train_mse']}")
  print(f"Average Testing MSE: {metrics['avg_test_mse']}, Std Testing MSE: {metrics['std_test_mse']}")
  print("\n")