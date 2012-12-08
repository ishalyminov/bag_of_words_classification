import copy

class FrequencyFilter(object):
    def __init__(self, in_frequency_chunks_number):
        self.frequency_chunks_number = in_frequency_chunks_number

    def load_distribution(self, in_distribution):
        self.distribution = copy.deepcopy(in_distribution)
        self.__slice_frequencies(self.distribution)

    def __slice_frequencies(self, in_distribution):
        uniq_frequencies = sorted(set(in_distribution.values()))
        freqs_number = len(uniq_frequencies)
        chunks_number = min(self.frequency_chunks_number, freqs_number)
        if not chunks_number:
            print '!'
        chunk_size = freqs_number / chunks_number
        # creating the specified number of chunks for frequency values
        frequency_intervals = range(0, freqs_number, chunk_size)[:self.frequency_chunks_number]
        frequency_intervals.append(len(uniq_frequencies) + 1)

        frequency_mapping = {}
        for index in xrange(len(frequency_intervals) - 1):
            (index_range_begin, index_range_end) = frequency_intervals[index : index + 2]
            for frequency in uniq_frequencies[index_range_begin : index_range_end]:
                frequency_mapping[frequency] = index

        self.frequency_chunks = [[]] * len(frequency_mapping.values())

        # at the end we have {'chunk': ['word1', 'word2', ...]} mapping
        for (word, frequency) in in_distribution.iteritems():
            index = frequency_mapping[frequency]
            self.frequency_chunks[index].append(word)

    # cut_head, cut_tail are the numbers of chunk to cut 
    # from the head and tail of the distribution respectively
    def get_filtered_distribution(self,
                                  cut_head = 0,
                                  cut_tail = None):
        result = {}
        if cut_tail:
            cut_tail = -cut_tail
        for chunk in self.frequency_chunks[cut_head : cut_tail]:
            for word in chunk:
                result[word] = self.distribution[word]
        return result

