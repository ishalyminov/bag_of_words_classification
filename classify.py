import sys
import os
import itertools
import numpy

import text_reading
import text_reading.twenty_newsgroups
import text_reading.ruscorpora
import dataset_loading
import bag_of_words
import classifier_wrapper
import frequency_chunking


def process_data(in_folder, in_sentences_extractor):
    bags = []
    (files, categories) = dataset_loading.get_files_list(in_folder)
    for filename in files:
        sentences = in_sentences_extractor(filename)
        bags.append(bag_of_words.sentences_to_bag_of_words(sentences))
    categories_dict = dataset_loading.get_categories_dict(categories)
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
    print >>sys.stderr, 'Train set size: %d' % len(train_answers)
    classifier.train(train_set_filtered, train_answers)

    test_set_filtered = []
    test_answers = []
    for (bag, answer) in zip(in_test_dataset.get_bags(), in_test_dataset.get_answers_vector()):
        bag_filtered = in_filter_policy.filter_distribution(bag)
        if len(bag_filtered):
            test_set_filtered.append(bag_filtered)
            test_answers.append(answer)
    print >>sys.stderr, 'Test set size: %d' % len(test_answers)

    answers = classifier.predict(test_set_filtered)
    return numpy.mean(answers == test_answers)

def perform_classification(in_train_data, in_test_data, in_dataset_type):
    sentences_extractor = None
    if in_dataset_type == '20newsgroups':
        sentences_extractor = getattr(text_reading.twenty_newsgroups, 'load_text')
    elif in_dataset_type == 'ruscorpora':
        sentences_extractor = getattr(text_reading.ruscorpora, 'get_text_raw')
    train_dataset = dataset_loading.DatasetLoader(in_train_data, sentences_extractor) #process_data(in_train_data, sentences_extractor)
    test_dataset = dataset_loading.DatasetLoader(in_test_data, sentences_extractor) # process_data(in_test_data, sentences_extractor)
    return classify_texts(train_dataset, test_dataset, frequency_chunking.FrequencyChunkFilter())

if __name__ == '__main__':
    if len(sys.argv) < 3:
        exit('Usage: classify.py <training data folder> <testing data root> [ruscorpora|20newsgroups = default]')
    dataset = '20newsgroups'
    if len(sys.argv) == 4:
        dataset = sys.argv[3]
    print perform_classification(sys.argv[1], sys.argv[2], dataset)
