import sys
import os
import itertools
import numpy

import bag_of_words
import twenty_newsgroups_reader
import classifier_wrapper
import frequency_filtering


def process_data(in_folder):
    bags = []
    (files, categories) = get_files_list(in_folder)
    for filename in files:
        bags.append(bag_of_words.read_file_into_map(filename))
    categories_dict = get_categories_dict(categories)
    categories_vector = [categories_dict[category] for category in categories]

    vectorizer = DictVectorizer()
    # builds a vocabulary out of all words in both sets
    term_document_matrix = vectorizer.fit_transform(bags)
    tfidf_transformer = TfidfTransformer()
    # in this matrix rows are documents, columns - features (terms' tfidf's)
    tfidf_matrix = tfidf_transformer.fit_transform(term_document_matrix)
    return (tfidf_matrix, categories_dict, categories_vector)

def perform_experiment(in_training_folder, in_testing_folder):
    train_set = twenty_newsgroups_reader.DatasetLoader(in_training_folder)
    test_set = twenty_newsgroups_reader.DatasetLoader(in_testing_folder)
    classifier = classifier_wrapper.ClassifierWrapper()

    freq_filter = frequency_filtering.FrequencyFilter(10)
    train_set_filtered = []
    for bag in train_set.get_bags():
        freq_filter.load_distribution(bag)
        train_set_filtered.append(freq_filter.get_filtered_distribution())
    classifier.train(train_set_filtered, train_set.get_answers_vector())

    test_set_filtered = []
    for bag in test_set.get_bags():
        freq_filter.load_distribution(bag)
        test_set_filtered.append(freq_filter.get_filtered_distribution())
    answers = classifier.predict(test_set_filtered)

    print numpy.mean(answers == test_set.get_answers_vector())

if __name__ == '__main__':
    if len(sys.argv) < 3:
        exit('Usage: classify.py <training data folder> <testing data root>')
    perform_experiment(sys.argv[1], sys.argv[2])
