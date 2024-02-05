## Homework 1

# Import necessary packages
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from math import sqrt
from math import log


# Training a Random Forest for each value of k
def train_random_forest(k, training_data, training_labels, test_data, test_labels):
  # Create a list to store the errors
  training_errors = []
  test_errors = []
  
  # Loop through each value of k
  for k in k_values:
    random_forest = RandomForestClassifier(
                    n_estimators=k, 
                    max_features = "sqrt",
                    random_state=15)

    random_forest.fit(training_data, training_labels)

    # Make predictions
    training_pred = random_forest.predict(training_data)
    test_pred = random_forest.predict(test_data)

    # Calculate misclassification errors
    training_error = 1 - accuracy_score(training_labels, training_pred)
    test_error = 1 - accuracy_score(test_labels, test_pred)

    # Add errors to error lists
    training_errors.append(training_error)
    test_errors.append(test_error)

  return training_errors, test_errors


# Training a Decision Tree up to depth 12
def train_decision_tree(training_data, test_data, training_labels, test_labels, depth):
  # Create a list to store the errors
  tree_depth = []
  training_errors = []
  test_errors = []

  for dep in range(1,depth+1):
    decision_tree = DecisionTreeClassifier(max_depth=dep)
    decision_tree.fit(training_data, training_labels)

  # Make predictions
    training_pred = decision_tree.predict(training_data)
    test_pred = decision_tree.predict(test_data)

  # Calulate miscalssification errors
    training_error = 1 - accuracy_score(training_labels, training_pred)
    test_error = 1 - accuracy_score(test_labels, test_pred)

  # Add errors to error lists and tree depth list
    training_errors.append(training_error)
    test_errors.append(test_error)
    tree_depth.append(dep)
    
  plot_errors_dt(training_errors, test_errors,tree_depth,)
  return training_errors, test_errors, tree_depth



def plot_errors_rf(training_errors, test_errors):
  # Plot the errors for a random forest
  plt.plot(k_values, training_errors, label="Train Error")
  plt.plot(k_values, test_errors, label="Test Error")
  plt.xlabel("Number of Trees (k)")
  plt.ylabel("Misclassifiaction Error")
  plt.legend()
  plt.show()


def plot_errors_dt(training_errors, test_errors, tree_depth):
  # Plot the errors for a decision tree
  plt.plot(tree_depth, training_errors, label="Training Error")
  plt.plot(tree_depth, test_errors, label="Test Error")
  plt.xlabel("Tree Depth")
  plt.ylabel("Misclassification Error")
  plt.legend()
  plt.show()


def error_dataframe_rf(k, training_errors, test_errors):
  # Create dataframe to display errors for a random forest
  errors_df_rf = pd.DataFrame({'Number of Trees (k)': k,
                          'Train Error': training_errors,
                          'Test Error': test_errors})
  print(errors_df_rf)

  # Convert errors dataframe to a csv file
  errors_df_rf.to_csv("errors_rf.csv")


def error_dataframe_dt(training_errors, test_errors, tree_depth):
  # Creating a dataframe to display the errors
  error_dataframe = pd.DataFrame(
    {'Tree Depth': tree_depth,
     'Training Error': training_errors,
     'Test Error': test_errors})
  error_dataframe.to_csv("errors_dt.csv")
  return error_dataframe


def find_minimum_test_error(error_dataframe):
  # Find the minimum test error
  min_test_error_row = error_dataframe.loc[error_dataframe['Test Error'].idxmin()]

  return min_test_error_row


  

file_loc = "/Users/kevin_smith/Desktop/FSU_Relevant_Stuff/spring23/STA5635/homework/hw1/data/MADELON"
training_data= pd.read_fwf(file_loc + "/madelon_train.data", header = None)
training_labels = pd.read_fwf(file_loc + "/madelon_train.labels", header = None)
test_data = pd.read_fwf(file_loc + "/madelon_valid.data", header = None)
test_labels = pd.read_fwf(file_loc + "/madelon_valid.labels", header = None)

# Create a list with k values (number of trees in the forest)
k_values = [3, 10, 30, 100, 300]

 # Running functions
training_errors, test_errors = train_random_forest(k_values, 
                     training_data,
                     training_labels,
                     test_data,
                     test_labels)
plot_errors_rf(training_errors, test_errors)
error_dataframe_rf(k_values, training_errors, test_errors)



## Running Decision Tree

# Get the data
file_loc = "/Users/kevin_smith/Desktop/FSU_Relevant_Stuff/spring23/STA5635/homework/hw1/data/MADELON"
training_data = pd.read_fwf(file_loc + "/madelon_train.data", header = None)
training_labels = pd.read_fwf(file_loc + "/madelon_train.labels", header = None)
test_data = pd.read_fwf(file_loc + "/madelon_valid.data", header = None)
test_labels = pd.read_fwf(file_loc + "/madelon_valid.labels", header = None)


# Running functions
max_tree_depth = 12
training_errors, test_errors, tree_depth = train_decision_tree(training_data, test_data, training_labels, test_labels, max_tree_depth)

#plot_errors_dt(training_errors, test_errors, tree_depth)
error_dataframe = error_dataframe_dt(training_errors, test_errors, tree_depth)
min_test_error_row = find_minimum_test_error(error_dataframe)
print(min_test_error_row)

