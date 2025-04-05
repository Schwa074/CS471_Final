import naivebayes as nb
import decisiontree as dt
import pandas as pd
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV


# Load the data
data = pd.read_csv('./data/onlinefraud.csv')

# Encode categorical variables to numeric values
label_encoder = preprocessing.LabelEncoder()
data['type'] = label_encoder.fit_transform(data['type'])  # Encode transaction type.
data['nameOrig'] = label_encoder.fit_transform(data['nameOrig'])  # Encode origin account ID.
data['nameDest'] = label_encoder.fit_transform(data['nameDest'])  # Encode destination account ID.

print(data.columns)
print(data.head())  # First 5 rows

features = data[['type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest']]
labels = data[['isFraud']]

# Split data into training (60%), validation (20%), and test (20%) sets
# First Split: 60% for training, 40% for further splitting into validation and test
train_features, remaining_features, train_labels, remaining_labels = train_test_split(features, labels, test_size=0.4, random_state=42)

# Second Split: 50% of the remaining data for validation and 50% for testing
validation_features, test_features, validation_labels, test_labels = train_test_split(remaining_features, remaining_labels, test_size=0.5, random_state=42)

# Use SMOTE to oversample the minority class and balance training data.
temp_features, temp_labels = SMOTE(random_state=130).fit_resample(train_features, train_labels)
train_features = temp_features
train_labels = temp_labels

print("Done")