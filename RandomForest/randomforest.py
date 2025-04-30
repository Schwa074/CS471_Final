import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV
import time
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

def run_random_forest(data_feature_names, data_target_names, train_features, train_labels, validation_features, validation_labels, test_features, test_labels):
    start_time = time.time()

    param_grid = {
    'n_estimators': [10, 20, 25, 30],
    'max_depth': np.arange(1, 11)
    }
    
    # To be more efficient, took best param to avoid doing every possible param
    # See RandomForestBestParamOutput for inital run
    # param_grid = {
    # 'n_estimators': [30],
    # 'max_depth': [10]
    # }

    # Perform Grid Search with cross-validation
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='f1')
    grid_search.fit(train_features, train_labels)

    # Plot hyperparameters vs performance (scores)
    results = grid_search.cv_results_
    mean_scores = results['mean_test_score']

    print(f"Min mean test score: {min(results['mean_test_score'])}")
    print(f"Max mean test score: {max(results['mean_test_score'])}")

    # See RandomForestHyperParamTune from inital run
    depth_range = np.arange(1, 11)  
    split_range = np.arange(2, 11)

    scores_matrix = results['mean_test_score'].reshape(len(depth_range), len(split_range))

    plt.figure(figsize=(8, 6))
    plt.imshow(scores_matrix, interpolation='nearest', cmap='viridis')
    plt.colorbar(label="Mean Test Score")
    plt.xticks(np.arange(len(split_range)), split_range)  # x-axis: min_samples_split
    plt.yticks(np.arange(len(depth_range)), depth_range)  # y-axis: max_depth
    plt.xlabel('min_samples_split')
    plt.ylabel('max_depth')
    plt.title('Grid Search Performance: Hyperparameters vs Accuracy')
    plt.show()

    # Train the model using the best hyperparameters found
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"Best params: {best_params}")
    print(f"Best recall from GridSearchCV: {best_score * 100:.2f}%")

    best_dt_path = './pickledData/bestRandomForestModel.pickle'  # Path to the pickled dataset file.

    # Load data from a pickle file or reading CSV file and pickle it if needed. Should only happen once
    try:
        with open(best_dt_path, "rb") as file:
            best_dt = pickle.load(file)
    except FileNotFoundError:
        best_dt = RandomForestClassifier(random_state=42, class_weight='recall', **best_params)
        best_dt.fit(np.concatenate((train_features, validation_features)), np.concatenate((train_labels, validation_labels)))
        with open(best_dt_path, "wb") as file:
            pickle.dump(best_dt, file)  # Save random forest model to a pickle for future use.

    # Feature importance
    important_features = best_dt.feature_importances_
    feature_importance_pairs = list(zip(data_feature_names, important_features))
    sorted_importances = sorted(feature_importance_pairs, key=lambda x: x[1], reverse=True)

    print("\nTop 10 Important Features in Random Forest:")
    for feature, importance in sorted_importances[:10]:
        print(f"{feature}: {importance:.4f}")

    # Plot top 10 feature importances
    top_features = sorted_importances[:10]
    features = [f[0] for f in top_features]
    importances = [f[1] for f in top_features]

    plt.figure(figsize=(10, 6))
    plt.barh(features, importances, color='orange')
    plt.xlabel('Feature Importance')
    plt.title('Top 10 Important Features - Random Forest')
    plt.gca().invert_yaxis()  # Highest on top
    plt.tight_layout()
    plt.show()

    # Predict on the training data
    train_predictions = best_dt.predict(train_features)

    # Generate a classification report for training data
    print("Random Forest Classification Report on Training Data:\n", classification_report(train_labels, train_predictions, target_names=['isNotFraud', 'isFraud']))

    # Generate a confusion matrix for training data
    print("Random Forest Confusion Matrix on Training Data:\n", confusion_matrix(train_labels, train_predictions))

    # Test the model on the test data
    test_predictions = best_dt.predict(test_features)

    # Generate a classification report
    print("Random Forest Classification Report on Testing Data:\n", classification_report(test_labels, test_predictions, target_names=['isNotFraud', 'isFraud']))

    print(f"Random Forest Confusion matrix on Testing Data:\n " + str(confusion_matrix(test_labels, test_predictions)))

    end_time = time.time()

    # Visualize and interpret the random forest
    plt.figure(figsize=(8, 5))
    plt.plot(depths, mean_scores, marker='o', color='green')
    plt.xlabel('Max Depth')
    plt.ylabel('Mean Recall Score')
    plt.title('Effect of max_depth on Recall')
    plt.grid(True)
    plt.show()

    print("--- Completed Random Forest evaluation ---")
    print(f"Execution time: {end_time - start_time:.4f} seconds\n")