import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

k_values = [3, 10, 30, 100, 300]
sqrt_features = int(np.sqrt(500))


file_loc = "/Users/kevin_smith/Desktop/FSU_Relevant_Stuff/spring23/STA5635/homework/hw1/data/MADELON"
train = pd.read_fwf(file_loc + "/madelon_train.data", header = None)
train_labels = pd.read_fwf(file_loc + "/madelon_train.labels", header = None)
test = pd.read_fwf(file_loc + "/madelon_valid.data", header = None)
test_labels = pd.read_fwf(file_loc + "/madelon_valid.labels", header = None)

# Combine features and labels
train['Class'] = train_labels
test['Class'] = test_labels

# e) All 500 features
train_errors_e = []
test_errors_e = []

for k in k_values:
    model = RandomForestClassifier(n_estimators=k, max_features=None)
    model.fit(train.drop('Class', axis=1), train['Class'])
    
    train_pred = model.predict(train.drop('Class', axis=1))
    test_pred = model.predict(test.drop('Class', axis=1))
    
    train_error = 1 - accuracy_score(train['Class'], train_pred)
    test_error = 1 - accuracy_score(test['Class'], test_pred)
    
    train_errors_e.append(train_error)
    test_errors_e.append(test_error)

# Plot errors
plt.plot(k_values, train_errors_e, label='Train Error (all)')
plt.plot(k_values, test_errors_e, label='Test Error (all)')
plt.xlabel('Number of Trees (k)')
plt.ylabel('Misclassification Error')
plt.legend()
plt.title('Random Forest Performance (All Features)')
plt.savefig("1e.png")
plt.show()

# Report errors in a table
table_e = pd.DataFrame({'K': k_values, 'Train Error (all)': train_errors_e, 'Test Error (all)': test_errors_e})
print("Table for e):")
print(table_e)
