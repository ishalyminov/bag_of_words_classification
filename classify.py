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

    freq_filter = frequency_filtering.FrequencyChunkFilter()
    train_set_filtered = []
    train_answers = []
    for (bag, answer) in zip(train_set.get_bags(), train_set.get_answers_vector()):
        freq_filter.load_distribution(bag, 10)
        bag_filtered = freq_filter.get_filtered_distribution(cut_tail = 1)
        if len(bag_filtered):
            train_set_filtered.append(bag_filtered)
            train_answers.append(answer)
    classifier.train(train_set_filtered, train_answers)

    test_set_filtered = []
    test_answers = []
    for (bag, answer) in zip(test_set.get_bags(), test_set.get_answers_vector()):
        freq_filter.load_distribution(bag, 10)
        bag_filtered = freq_filter.get_filtered_distribution(cut_tail = 1)
        if len(bag_filtered):
            test_set_filtered.append(bag_filtered)
            test_answers.append(answer)
    answers = classifier.predict(test_set_filtered)

    print numpy.mean(answers == test_answers)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        exit('Usage: classify.py <training data folder> <testing data root>')
    perform_experiment(sys.argv[1], sys.argv[2])
