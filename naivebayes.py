# import pandas as pd

# # Load the dataset
# df = pd.read_csv('SpamDetection.csv') # TODO Change csv

# # Split into training and testing sets
# training_data = df[:20]
# testing_data = df[20:]

# # Calculate prior probabilities (P(spam) and P(ham)) based on the training data
# spam_count = len(training_data[training_data['Target'] == 'spam']) # TODO Change target value and variable names
# ham_count = len(training_data[training_data['Target'] == 'ham']) # TODO Change target value and variable names

# total_train_messages = len(training_data)

# P_spam = spam_count / total_train_messages
# P_ham = ham_count / total_train_messages

# # Create word count dictionaries for spam and ham classes based on the training data
# spam_words = ' '.join(training_data[training_data['Target'] == 'spam']['data']).split() # TODO Change target value and variable names
# ham_words = ' '.join(training_data[training_data['Target'] == 'ham']['data']).split() # TODO Change target value and variable names

# # Create dictionaries to store word counts
# spam_word_count = {word: spam_words.count(word) for word in set(spam_words)} # TODO Change variable names
# ham_word_count = {word: ham_words.count(word) for word in set(ham_words)} # TODO Change variable names

# # Vocabulary - all unique words across both classes
# vocabulary = set(spam_words + ham_words) # TODO Change variable names

# # Calculate total word counts in spam and ham classes
# spam_total = sum(spam_word_count.values()) # TODO Change variable names
# ham_total = sum(ham_word_count.values()) # TODO Change variable names

# # Laplace smoothing: adding 1 to the word count and the size of the vocabulary to the denominator
# def perform_laplace_smoothing(word, word_count, total_count):
#     return (word_count.get(word, 0) + 1) / (total_count + len(vocabulary))

# # Function to classify a sentence
# def classify_sentence(sentence):
#     # Normalize the sentence and split each word
#     test_words = sentence.lower().split()
    
#     P_test_given_spam = P_spam
#     P_test_given_ham = P_ham

#     for word in test_words:
#         P_test_given_spam *= perform_laplace_smoothing(word, spam_word_count, spam_total)
#         P_test_given_ham *= perform_laplace_smoothing(word, ham_word_count, ham_total)

#     # Print the posterior probabilities
#     print(f"Posterior Probability of Spam: {P_test_given_spam:.8e}")
#     print(f"Posterior Probability of Ham: {P_test_given_ham:.8e}")

#     if P_test_given_spam > P_test_given_ham:
#         return 'spam'
#     else:
#         return 'ham'

# correct_predictions = 0

# for index, row in testing_data.iterrows():
#     # Grab the data and target from the testing data set
#     sentence = row['data']
#     actual_classification = row['Target']

#     # Print out results of the classification

#     print(f"Sentence: {sentence}")

#     # Classify the sentence and return either 'spam' or 'ham' - will also print out posterior probability
#     predicted_classification = classify_sentence(sentence)
    
#     # Keep track of correct predictions for average
#     if predicted_classification == actual_classification:
#         correct_predictions += 1
    
#     print(f"Predicted Classification: {predicted_classification}")
#     print(f"Actual Classification: {actual_classification}")

#     print("-" * 50)

# # Calculate accuracy
# accuracy = correct_predictions / len(testing_data)
# print(f"Accuracy: {accuracy:.2f}")

#  # TODO Add full Classification Report via sklearn.metrics

