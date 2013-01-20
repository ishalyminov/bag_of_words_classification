import os
import sys
import collections
import nltk.corpus
import operator
import itertools

import text_reading
import text_reading.ruscorpora
import text_reading.twenty_newsgroups
import dataset_loading
import matplotlib.pyplot as plot

def get_sentences_extractor(in_dataset_name):
    if in_dataset_name == '10newspapers_ruscorpora':
        return getattr(text_reading.ruscorpora, 'get_text_raw')
    elif in_dataset_name == '20newsgroups':
        return getattr(text_reading.twenty_newsgroups, 'load_text_raw')

def get_stop_list(in_dataset_name):
    if in_dataset_name == '10newspapers_ruscorpora':
        return [word.decode('utf-8') for word in nltk.corpus.stopwords.words('russian')]
    elif in_dataset_name == '20newsgroups':
        return [word for word in nltk.corpus.stopwords.words('english')]

def process_folder(in_folder):
    stopwords_distribution = []
    words_distribution = []
    stopwords_found = set([])
    sentence_extractor = get_sentences_extractor(sys.argv[2])
    stop_list = get_stop_list(sys.argv[2])
    dataset = dataset_loading.DatasetLoader(in_folder, sentence_extractor)
    for bag in dataset.get_bags():
        freqs_sorted = sorted(set(bag.values()), reverse = True)
        freqs_to_ranks = {freq: rank for (freq, rank) in zip(freqs_sorted, itertools.count())}
        for term, frequency in bag.iteritems():
            rank = freqs_to_ranks[frequency]
            if term in stop_list:
                stopwords_found.add(term)
                stopwords_distribution.append(rank)
            else:
                words_distribution.append(rank)
    plot.hist(stopwords_distribution, alpha=0.5, bins=50, color='red', label='stopwords')
    plot.hist(words_distribution, alpha=0.5, bins=50, color='green', label='words')
    plot.title('Words/stopwords distribution for "%s" texts' % sys.argv[2])
    plot.grid(True)
    plot.legend()
    plot.savefig(sys.argv[3])
    plot.clf()

if __name__ == '__main__':
    if len(sys.argv) < 4:
        exit('Usage: stop_words_distribution.py <source folder> <dataset name> <output file name>')
    process_folder(os.path.abspath(sys.argv[1]))
