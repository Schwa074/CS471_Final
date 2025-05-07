import DecisionTree.decisiontree as dt
import RandomForest.randomforest as rf
import NaiveBayes.naivebayes as nb
import pandas as pd
import pickle
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
import time
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, plot_tree
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Load the data TAKES WAY TO LONG FOR 6 million records
# data = pd.read_csv('./data/onlinefraud.csv')
start_time = time.time()

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
labels = data['isFraud']

# Split data into training (60%), validation (20%), and test (20%) sets
# First Split: 60% for training, 40% for further splitting into validation and test
train_features, remaining_features, train_labels, remaining_labels = train_test_split(features, labels, test_size=0.4, random_state=42)

# Second Split: 50% of the remaining data for validation and 50% for testing
validation_features, test_features, validation_labels, test_labels = train_test_split(remaining_features, remaining_labels, test_size=0.5, random_state=42)

# Plot class distribution before SMOTE
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.countplot(x=train_labels, palette="coolwarm")
plt.title("Class Distribution Before SMOTE")
plt.xlabel("Is Fraud")
plt.ylabel("Count (millions)")

# Use SMOTE to balance training data.
temp_features, temp_labels = SMOTE(random_state=130).fit_resample(train_features, train_labels)
train_features = temp_features
train_labels = temp_labels

# Plot class distribution after SMOTE
plt.subplot(1, 2, 2)
sns.countplot(x=temp_labels, palette="coolwarm")
plt.title("Class Distribution After SMOTE")
plt.xlabel("Is Fraud")
plt.ylabel("Count (millions)")

plt.tight_layout()
plt.show()

nb.run_naive_bayes(features, labels, train_features, train_labels, validation_features, validation_labels, test_features, test_labels)

dt.run_decision_tree(features, labels, train_features, train_labels, validation_features, validation_labels, test_features, test_labels)

rf.run_random_forest(features, labels, train_features, train_labels, validation_features, validation_labels, test_features, test_labels)

end_time = time.time()

print("\n --- All algorithms completed ---")
print(f"Execution time of all algorithms: {end_time - start_time:.4f} seconds")