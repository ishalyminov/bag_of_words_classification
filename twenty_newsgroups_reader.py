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
    def __init__(self, in_texts_root):
        self.texts_root = in_texts_root
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
            bag = bag_of_words.read_file_into_map(filename)
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
