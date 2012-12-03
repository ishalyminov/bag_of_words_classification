import sys
import os
import bag_of_words
import itertools
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC


def get_files_list(in_root_folder):
    result_files = []
    result_categories = []
    for root, folders, files in os.walk(in_root_folder, followlinks = True):
        for filename in files:
            result_files.append(os.path.join(root, filename))
            result_categories.append(os.path.split(root)[1])
    return (result_files, result_categories)

def get_categories_dict(in_categories_list):
    categories_dict = {}
    uniq_categories = set(in_categories_list)
    for category, index in zip(uniq_categories, itertools.count()):
        categories_dict[category] = index
    return categories_dict

def do_classification(in_train_bags, in_test_bags, in_train_answers, in_test_answers):
    vectorizer = DictVectorizer()
    # builds a vocabulary out of all words in both sets
    vectorizer.fit(in_train_bags + in_test_bags)
    term_document_matrix = vectorizer.transform(train_bags)
    tfidf_transformer = TfidfTransformer()
    # in this matrix rows are documents, columns - features (terms' tfidf's)
    tfidf_matrix = tfidf_transformer.fit_transform(term_document_matrix)
    classifier_answers = [categories_dict[category] for category in in_train_answers]

    classifier = SVC()
    classifier.fit(tfidf_matrix, classifier_answers)
    classifier_predictions = classifier.predict()

def perform_experiment(in_training_folder, in_testing_folder):
    train_bags= []
    test_bags = []
    (train_files, train_categories) = get_files_list(in_training_folder)
    (test_files, test_categories) = get_files_list(in_testing_folder)
    for filename in train_files:
        train_bags.append(bag_of_words.read_file_into_map(filename))
    for filename in test_files:
        test_bags.append(bag_of_words.read_file_into_map(filename))
    do_classification(test_bags, train_bags, train_categories, test_categories)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        exit('Usage: classify.py <training data folder> <testing data root>')
    perform_experiment(sys.argv[1], sys.argv[2])
