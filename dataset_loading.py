import string
import sys
import nltk.tokenize
import re
import os
import itertools

import bag_of_words

def get_files_list(in_root_folder):
    result_files = []
    result_categories = []
    for root, folders, files in os.walk(in_root_folder, followlinks = True):
        for filename in files:
            result_files.append(os.path.join(root, filename))
            result_categories.append(os.path.basename(root))
    return (result_files, result_categories)

def get_categories_dict(in_categories_list):
    categories_dict = {}
    uniq_categories = set(in_categories_list)
    for category, index in zip(uniq_categories, itertools.count()):
        categories_dict[category] = index
    return categories_dict

class DatasetLoader(object):
    def __init__(self, in_texts_root, in_sentences_extractor, in_stop_list = []):
        self.texts_root = in_texts_root
        self.sentences_extractor = in_sentences_extractor
        self.stop_list = in_stop_list
        (files, categories) = get_files_list(self.texts_root)
        all_categories = categories
        self.categories_dict = get_categories_dict(categories)
        (self.bags, indices) = self.__make_dataset_bags(files)
        self.categories = [categories[index] for index in indices]
        #self.full_dictionary = self.__build_full_dictionary(self.bags)

    def __make_dataset_bags(self, in_files):
        result = []
        file_indices = []
        for (filename, index) in zip(in_files, itertools.count()):
            sentences = self.sentences_extractor(filename)
            sentences_filtered = []
            for raw_sentence in sentences:
                sentences_filtered.append([word for word in raw_sentence if word not in self.stop_list])
            bag = bag_of_words.sentences_to_bag_of_words(sentences_filtered)
            if len(bag) >= 20:
                result.append(bag)
                file_indices.append(index)
        return (result, file_indices)

    def __build_full_dictionary(self, in_bags):
        result = {}
        for bag in in_bags:
            result.update(in_bags.items())
        return result


    def get_bags(self):
        return self.bags

    def get_answers_vector(self):
        return [self.categories_dict[category] for category in self.categories]

if __name__ == '__main__':
    if len(sys.argv) < 2:
        exit('usage: 20news_reader.py <text file>')
    print load_file(sys.argv[1])
