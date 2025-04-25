import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV
import time
from sklearn.tree import DecisionTreeClassifier, plot_tree
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

def run_decision_tree(data_feature_names, data_target_names, train_features, train_labels, validation_features, validation_labels, test_features, test_labels):
    start_time = time.time()

    param_grid = {
        'max_depth': np.arange(1, 11),  # Depth of the tree (1 to 10)
        'min_samples_split': np.arange(2, 11)  # Minimum samples to split (2 to 10)
    }

    # To be more efficient, took best param to avoid doing every possible param
    # See DecisionTreeBestParamOutput for inital run
    # param_grid = {
    #     'max_depth': [10],  
    #     'min_samples_split': [2]
    # }

    # Perform Grid Search with cross-validation
    grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='recall')
    grid_search.fit(train_features, train_labels)

    # Plot hyperparameters vs performance (scores)
    results = grid_search.cv_results_
    print(f"Min mean test score: {min(results['mean_test_score'])}")
    print(f"Max mean test score: {max(results['mean_test_score'])}")

    See DecisionTreeHyperParamTune from inital run
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

    best_dt_path = './pickledData/bestDecisionTreeModel.pickle'  # Path to the pickled dataset file.

    # Load data from a pickle file or reading CSV file and pickle it if needed. Should only happen once
    try:
        with open(best_dt_path, "rb") as file:
            best_dt = pickle.load(file)
    except FileNotFoundError:
        best_dt = DecisionTreeClassifier(random_state=42, **best_params)
        best_dt.fit(np.concatenate((train_features, validation_features)), np.concatenate((train_labels, validation_labels)))
        with open(best_dt_path, "wb") as file:
            pickle.dump(best_dt, file)  # Save decision tree model to a pickle for future use.

    # Evaluate on training set
    train_predictions = best_dt.predict(train_features)

    # Classification report for training set
    print("Decision Tree Classification Report on Training Set:\n",
        classification_report(train_labels, train_predictions, target_names=['isNotFraud', 'isFraud']))

    # Confusion matrix for training set
    print("Decision Tree Confusion Matrix on Training Set:\n",
        confusion_matrix(train_labels, train_predictions))

    # Test the model on the test data
    test_predictions = best_dt.predict(test_features)

    # Generate a classification report
    print("Decision Tree Classification Report on Testing Set:\n", classification_report(test_labels, test_predictions, target_names=['isNotFraud', 'isFraud']))

    print(f"Decision Tree Confusion matrix on Testing Set:\n " + str(confusion_matrix(test_labels, test_predictions)))

    end_time = time.time()

    print("--- Completed Decision Tree evaluation ---")
    print(f"Execution time: {end_time - start_time:.4f} seconds\n")

    # Visualize and interpret the decision tree - Not really viewable since it has a depth of 10 but it's under the images dir
    plt.figure(figsize=(20, 10))  # Create a figure and set size
    plot_tree(
        best_dt,
        filled=True,
        feature_names=list(data_feature_names),
        class_names=[str(name) for name in data_target_names],
        fontsize=10
    )
    plt.suptitle("Decision Tree Visualization", fontsize=16)  # Use suptitle here
    plt.savefig("images/decision_tree.png", dpi=300, bbox_inches='tight')
    plt.show()