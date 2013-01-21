import copy
import numpy

# makes "in_chunks_number" equally long intervals
# for the input values range 
def slice_range(in_range, in_chunks_number):
    if not len(in_range):
        return []
    (min_val, max_val) = (min(in_range), max(in_range))
    if max_val == min_val:
        return [min_val]
    slicing_step = (max_val - min_val) / float(in_chunks_number)
    interval_begins = numpy.arange(min_val, max_val, slicing_step)
    # each interval's begin is the previous interval's end
    return interval_begins

class FrequencyRangePartitioner(object):
    def process_distribution(self, in_distribution, in_chunks_number):
        self.distribution = in_distribution
        self.frequency_interval_begins = slice_range(self.distribution.values(), in_chunks_number)
        self.words_split = [[] for index in xrange(in_chunks_number)]
        self.frequencies_split = [[] for index in xrange(in_chunks_number)]
        for word, frequency in self.distribution.iteritems():
            interval_index = self.__find_interval(frequency, self.frequency_interval_begins)
            self.words_split[interval_index].append(word)
            self.frequencies_split[interval_index].append(frequency)

    def __find_interval(self, in_value, in_intervals):
        for index in xrange(len(in_intervals) - 1):
            if in_value >= in_intervals[index] and in_value < in_intervals[index + 1]:
                return index
        return len(in_intervals) - 1

    def get_frequency_intervals(self):
        frequencies = self.distribution
        min_val, max_val = (min(self.distribution), max(self.distribution))
        result = \
            [(self.frequency_interval_begins[index], self.frequency_interval_begins[index + 1]) \
             for index in xrange(len(self.frequency_interval_begins) - 1)]
        result.append((frequency_interval_begins[-1], max_val))
        return result

    def get_partitioned_frequencies(self):
        return self.frequencies_split

    def get_partitioned_words(self):
        return self.words_split

class FrequencyChunkFilter(object):
    def __init__(self):
        self.partitioner = FrequencyRangePartitioner()

    def load_distribution(self, in_distribution, in_chunks_number):
        self.distribution = copy.deepcopy(in_distribution)
        self.partitioner.process_distribution(in_distribution, in_chunks_number)

    # cut_head, cut_tail are the numbers of chunk to cut 
    # from the head and tail of the distribution respectively
    def get_filtered_distribution(self, cut_head = 0, cut_tail = 0):
        result = {}
        if cut_head == 0:
            cut_head = None
        else:
            cut_head = -cut_head
        # distribution tail == low frequencies == partitioning begin
        # distribution tail == high frequencies == partitioning end
        (cut_left, cut_right) = (cut_tail, cut_head)
        for words_chunk in self.partitioner.get_partitioned_words()[cut_left : cut_right]:
            for word in words_chunk:
                result[word] = self.distribution[word]
        return result

class FrequencyGroupFilter(object):
    def load_distribution(self, in_distribution):
        self.distribution = copy.deepcopy(in_distribution)
        self.frequency_groups = sorted(set(self.distribution.values()), reverse = True)

    def get_filtered_distribution(self, cut_head = 0, cut_tail = 0):
        if cut_tail == 0:
            cut_tail = None
        else:
            cut_tail = -cut_tail
        filtered_frequency_groups = set(self.frequency_groups[cut_head:cut_tail])
        result = {word: frequency for (word, frequency) in self.distribution.iteritems() \
                  if frequency in filtered_frequency_groups}
        return result

