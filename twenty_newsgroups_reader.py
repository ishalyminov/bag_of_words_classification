import string
import sys
import nltk.tokenize
import re
import os
import itertools

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import bag_of_words

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

class DatasetLoader(object):
    def __init__(self, in_train_folder, in_test_folder):
        self.train_root = in_train_folder
        self.test_root = in_test_folder
        tfidf_transformer = TfidfTransformer()

        (train_files, train_categories) = get_files_list(in_train_folder)
        (test_files, test_categories) = get_files_list(in_test_folder)

        self.answers = {'train': train_categories, 'test': test_categories}
        self.categories_dict = get_categories_dict(train_categories + test_categories)
        train_bags = self.make_dataset_bags(train_files)
        test_bags = self.make_dataset_bags(test_files)

        vectorizer = self.prepare_vectorizer(train_bags, test_bags)

        term_doc_train = vectorizer.transform(train_bags)
        term_doc_test = vectorizer.transform(test_bags)

        self.tfidf_matrices = {'train': tfidf_transformer.fit_transform(term_doc_train),
                               'test': tfidf_transformer.fit_transform(term_doc_test)}

    def make_dataset_bags(self, in_files):
        return [bag_of_words.read_file_into_map(filename) for filename in in_files]

    def prepare_vectorizer(self, in_train_bags, in_test_bags):
        features = {}
        for bag in in_train_bags + in_test_bags:
            features.update(bag)
        vectorizer = DictVectorizer()
        vectorizer.fit(features)
        return vectorizer

    def get_term_doc_matrix(self, in_set_name):
        return self.tfidf_matrices[in_set_name]

    def get_answers_vector(self, in_set_name):
        answers = self.answers[in_set_name]
        return [self.categories_dict[category] for category in answers]


# Returns lists of word tokens free of punctuation marks
def load_text(in_file_name):
    in_file = open(in_file_name)
    from_line = in_file.readline().strip()
    subject_line = in_file.readline().strip()
    organization_line = in_file.readline().strip()
    lines_number_line = in_file.readline().strip()

    text = []
    for line in in_file:
        # for various email quotations
        line = line.lstrip(string.punctuation)
        text.append(line.strip())
    # tokenized punctuation-free sentences
    result = []
    for sentence in nltk.sent_tokenize(' '.join(text)):
        result.append([word.lower() for word in nltk.word_tokenize(sentence) \
                       if not re.match('^[^\w]+$', word)])
    return result



if __name__ == '__main__':
    if len(sys.argv) < 2:
        exit('usage: 20news_reader.py <text file>')
    print load_file(sys.argv[1])
