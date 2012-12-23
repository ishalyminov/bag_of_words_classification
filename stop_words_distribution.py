import os
import sys
import collections
import nltk.corpus

import twenty_newsgroups_reader
import matplotlib.pyplot as plot

def process_folder(in_folder):
    stopwords_distribution = []
    words_distribution = []
    for root, dirs, files in os.walk(in_folder, followlinks = True):
        for file_name in files:
            bag = collections.defaultdict(lambda: 0)
            sentences = twenty_newsgroups_reader.load_text(os.path.join(root, file_name))
            for sentence in sentences:
                for word in sentence:
                    bag[word.lower()] += 1
            frequencies_sorted = sorted(set(bag.values()), reverse = True)
            freq_ranks = {frequencies_sorted[index] : index \
                          for index in xrange(len(frequencies_sorted))}
            for (word, frequency) in bag.iteritems():
                rank = freq_ranks[frequency]
                if word in nltk.corpus.stopwords.words('english'):
                    stopwords_distribution.append(rank)
                else:
                    words_distribution.append(rank)

    plot.hist(stopwords_distribution, alpha=0.5, bins=50, color='red', label='stopwords')
    plot.hist(words_distribution, alpha=0.5, bins=50, color='green', label='words')
    plot.title('Words/stopwords ranks distribution in "20newsgroups" texts')
    plot.grid(True)
    plot.legend()
    plot.savefig('./stop_words_distribution.png')
    plot.clf()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        exit('Usage: stop-words_distribution.py <source folder>')
    process_folder(os.path.abspath(sys.argv[1]))