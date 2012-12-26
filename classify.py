import sys
import os
import itertools
import numpy

import bag_of_words
import twenty_newsgroups_reader
import classifier_wrapper
import frequency_chunking


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

def classify_texts(in_train_dataset,
                   in_test_dataset,
                   in_filter_policy):
    classifier = classifier_wrapper.ClassifierWrapper()

    train_set_filtered = []
    train_answers = []
    for (bag, answer) in zip(in_train_dataset.get_bags(), in_train_dataset.get_answers_vector()):
        bag_filtered = in_filter_policy.filter_distribution(bag)
        if len(bag_filtered):
            train_set_filtered.append(bag_filtered)
            train_answers.append(answer)
    classifier.train(train_set_filtered, train_answers)

    test_set_filtered = []
    test_answers = []
    for (bag, answer) in zip(in_test_dataset.get_bags(), in_test_dataset.get_answers_vector()):
        bag_filtered = in_filter_policy.filter_distribution(bag)
        if len(bag_filtered):
            test_set_filtered.append(bag_filtered)
            test_answers.append(answer)

    answers = classifier.predict(test_set_filtered)
    return numpy.mean(answers == test_answers)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        exit('Usage: classify.py <training data folder> <testing data root>')
    # print classify_texts(sys.argv[1], sys.argv[2], frequency_chunking.FrequencyChunkFilter())
