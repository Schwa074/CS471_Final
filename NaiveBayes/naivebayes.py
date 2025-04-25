import pandas as pd
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
import time
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def run_naive_bayes(data_feature_names, data_target_names, train_features, train_labels, validation_features, validation_labels, test_features, test_labels):
    start_time = time.time()

    # Since Gaussian Naive Bayes has fewer hyperparameters, we'll mainly tune var_smoothing
    param_grid = {
        'var_smoothing': np.logspace(0, -9, num=10)  # Test different smoothing values
    }

    # Perform Grid Search with cross-validation
    grid_search = GridSearchCV(GaussianNB(), param_grid, cv=5, scoring='recall')
    grid_search.fit(train_features, train_labels)

    # Get results
    results = grid_search.cv_results_
    smoothing_values = results['param_var_smoothing'].data.astype(np.float64)
    mean_scores = results['mean_test_score']
    print(f"Min mean test score: {min(results['mean_test_score'])}")
    print(f"Max mean test score: {max(results['mean_test_score'])}")

    plt.figure(figsize=(8, 5))
    plt.semilogx(smoothing_values, mean_scores, marker='o', color='teal')
    plt.title("Hyperparameter Tuning: var_smoothing vs Recall")
    plt.xlabel("var_smoothing (log scale)")
    plt.ylabel("Mean Cross-Validated Recall")
    plt.grid(True)
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.show()

    # Get best parameters and score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"Best params: {best_params}")
    print(f"Best recall from GridSearchCV: {best_score * 100:.2f}%")

    best_nb_path = './pickledData/bestNaiveBayesModel.pickle'

    # Load or create the best model
    try:
        with open(best_nb_path, "rb") as file:
            best_nb = pickle.load(file)
    except FileNotFoundError:
        best_nb = GaussianNB(**best_params)
        best_nb.fit(np.concatenate((train_features, validation_features)), 
                   np.concatenate((train_labels, validation_labels)))
        with open(best_nb_path, "wb") as file:
            pickle.dump(best_nb, file)

    # Training set evaluation
    train_predictions = best_nb.predict(train_features)
    print("Naive Bayes Classification Report on Training Set:\n", 
        classification_report(train_labels, train_predictions, target_names=['isNotFraud', 'isFraud']))

    print(f"Naive Bayes Confusion matrix on Training Set:\n {confusion_matrix(train_labels, train_predictions)}\n \n ")

    # Test the model
    test_predictions = best_nb.predict(test_features)

    # Generate classification report
    print("Naive Bayes Classification Report on Testing Set:\n", 
          classification_report(test_labels, test_predictions, 
                             target_names=['isNotFraud', 'isFraud']))

    print(f"Naive Bayes Confusion matrix on Testing Set:\n {confusion_matrix(test_labels, test_predictions)}\n \n ")

    end_time = time.time()
    print("--- Completed Naive Bayes evaluation ---")
    print(f"Execution time: {end_time - start_time:.4f} seconds\n")