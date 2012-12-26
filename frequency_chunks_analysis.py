import os
import sys
import collections
import operator
import numpy

import twenty_newsgroups_reader
import frequency_filtering

def get_categorized_statistics(in_texts_root):
    for category in os.listdir(in_texts_root):
        yield (category, get_statistics(os.path.join(in_texts_root, category)))

def get_overall_statistics(in_texts_root):
    return [('overall', get_statistics(in_texts_root))]

def get_statistics(in_texts_root):
    frequency_chunks_dict = collections.defaultdict(lambda: [])
    documents_set = twenty_newsgroups_reader.DatasetLoader(in_texts_root)
    partitioner = frequency_filtering.FrequencyRangePartitioner()

    for (bag, answer) in zip(documents_set.get_bags(), documents_set.get_answers_vector()):
        partitioner.process_distribution(bag, 20)
        frequency_chunks = partitioner.get_partitioned_frequencies()
        for index in xrange(len(frequency_chunks)):
            frequency_chunks_dict[index] += frequency_chunks[index]
    frequency_chunks_statistics = []
    for (chunk, frequencies) in sorted(frequency_chunks_dict.items(), key = operator.itemgetter(0)):
        frequency_chunks_statistics.append((numpy.mean(frequencies), numpy.std(frequencies)))
    return frequency_chunks_statistics


if __name__ == '__main__':
    if len(sys.argv) < 2:
        exit('Usage: frequency_chunks_analysis.py <texts root folder>')
    for chunk_stats in get_overall_statistics(os.path.abspath(sys.argv[1])):
        print '================\n%s' % chunk_stats[0]
        for index in xrange(len(chunk_stats[1])):
            chunk = chunk_stats[1][index]
            print 'Frequency chunk #%d: Mean(freq) = %d, StdDev(freq) = %d' % \
                  (index, chunk[0], chunk[1])
