import naivebayes as nb
import decisiontree as dt
import pandas as pd
import pickle
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.tree import DecisionTreeClassifier, plot_tree
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


# Load the data TAKES WAY TO LONG FOR 6 million records
# data = pd.read_csv('./data/onlinefraud.csv')

data_path = './pickledData/fraudData.pickle'  # Path to the pickled dataset file.

# Load data from a pickle file or reading CSV file and pickle it if needed. Should only happen once
try:
    with open(data_path, "rb") as file:
        data = pickle.load(file)
except FileNotFoundError:
    data = pd.read_csv('./data/onlinefraud.csv')  # Load dataset from CSV if pickle not found.
    with open(data_path, "wb") as fp:
        pickle.dump(data, fp)  # Save dataset to a pickle for future use.

# Encode categorical variables to numeric values
label_encoder = preprocessing.LabelEncoder()
data['type'] = label_encoder.fit_transform(data['type'])  # Encode transaction type.
data['nameOrig'] = label_encoder.fit_transform(data['nameOrig'])  # Encode origin account ID.
data['nameDest'] = label_encoder.fit_transform(data['nameDest'])  # Encode destination account ID.

print(data.columns)
print(data.head())  # First 5 rows

print(data.shape)

features = data[['type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest']]
labels = data[['isFraud']]

# Split data into training (60%), validation (20%), and test (20%) sets
# First Split: 60% for training, 40% for further splitting into validation and test
train_features, remaining_features, train_labels, remaining_labels = train_test_split(features, labels, test_size=0.4, random_state=42)

# Second Split: 50% of the remaining data for validation and 50% for testing
validation_features, test_features, validation_labels, test_labels = train_test_split(remaining_features, remaining_labels, test_size=0.5, random_state=42)

# Use SMOTE to balance training data.
temp_features, temp_labels = SMOTE(random_state=130).fit_resample(train_features, train_labels)
train_features = temp_features
train_labels = temp_labels

# # Decision Tree
# dt = DecisionTreeClassifier(random_state=42)
# dt.fit(train_features, train_labels)

# TODO Pickle the dt
dt_path = './pickledData/decisionTreeModel.pickle'  # Path to the pickled dataset file.

# Load data from a pickle file or reading CSV file and pickle it if needed. Should only happen once
try:
    with open(dt_path, "rb") as file:
        dt = pickle.load(file)
except FileNotFoundError:
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(train_features, train_labels)
    with open(dt_path, "wb") as file:
        pickle.dump(dt, file)  # Save decision tree model to a pickle for future use.

param_grid = {
    'max_depth': np.arange(1, 11),  # Depth of the tree (1 to 10)
    'min_samples_split': np.arange(2, 11)  # Minimum samples to split (2 to 10)
}



# Perform Grid Search with cross-validation
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(train_features, train_labels)

# Plot hyperparameters vs performance (scores)
results = grid_search.cv_results_
print(f"Min mean test score: {min(results['mean_test_score'])}")
print(f"Max mean test score: {max(results['mean_test_score'])}")

# depth_range = np.arange(1, 11)  
# split_range = np.arange(2, 11)

# scores_matrix = results['mean_test_score'].reshape(len(depth_range), len(split_range))

# plt.figure(figsize=(8, 6))
# plt.imshow(scores_matrix, interpolation='nearest', cmap='viridis')
# plt.colorbar(label="Mean Test Score")
# plt.xticks(np.arange(len(split_range)), split_range)  # x-axis: min_samples_split
# plt.yticks(np.arange(len(depth_range)), depth_range)  # y-axis: max_depth
# plt.xlabel('min_samples_split')
# plt.ylabel('max_depth')
# plt.title('Grid Search Performance: Hyperparameters vs Accuracy')
# plt.show()

# # Train the model using the best hyperparameters found
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"Best params: {best_params}")
print(f"Best recall from GridSearchCV: {best_score * 100:.2f}%")

# TODO Pickle the dt
best_dt_path = './pickledData/bestDecisionTreeModel.pickle'  # Path to the pickled dataset file.

# Load data from a pickle file or reading CSV file and pickle it if needed. Should only happen once
try:
    with open(best_dt_path, "rb") as file:
        best_dt = pickle.load(file)
except FileNotFoundError:
    best_dt = DecisionTreeClassifier(random_state=42, **best_params)
    best_dt.fit(np.concatenate((train_features, validation_features)), np.concatenate((train_labels, validation_labels)))
    with open(best_dt_path, "wb") as file:
        pickle.dump(dt, file)  # Save decision tree model to a pickle for future use.

# best_dt = DecisionTreeClassifier(random_state=42, **best_params)
# best_dt.fit(np.concatenate((train_features, validation_features)), np.concatenate((train_labels, validation_labels)))

# # Test the model on the test data
# test_predictions = best_dt.predict(test_features)

# # Generate a classification report
# print("Classification Report:\n", classification_report(test_labels, test_predictions))

# # Visualize and interpret the decision tree
# plt.figure(figsize=(12, 8))
# plot_tree(best_dt, filled=True, feature_names=data.feature_names, class_names=data.target_names, fontsize=10)
# plt.title("Decision Tree Visualization")
# plt.show()

print("Done")