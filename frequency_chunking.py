import frequency_filtering

DEFAULT_CHUNKS_NUMBER = 20

class FrequencyChunkFilterWrapper(object):
    def __init__(self, 
                 in_chunks_number = DEFAULT_CHUNKS_NUMBER,
                 in_cut_head = 0,
                 in_cut_tail = 0):
        self.chunks_number = in_chunks_number
        self.cut_head = in_cut_head
        self.cut_tail = in_cut_tail

    def filter_distribution(self, in_distribution):
        freq_filter = frequency_filtering.FrequencyChunkFilter()
        freq_filter.load_distribution(in_distribution, self.chunks_number)
        bag_filtered = freq_filter.get_filtered_distribution(cut_tail = self.cut_tail,
                                                             cut_head = self.cut_head)
        return bag_filtered

class FrequencyGroupFilterWrapper(object):
    def __init__(self,
                 in_cut_head = 0,
                 in_cut_tail = 0):
        self.cut_head = in_cut_head
        self.cut_tail = in_cut_tail

    def filter_distribution(self, in_distribution):
        freq_filter = frequency_filtering.FrequencyGroupFilter()
        freq_filter.load_distribution(in_distribution)
        bag_filtered = freq_filter.get_filtered_distribution(cut_tail = self.cut_tail,
                                                             cut_head = self.cut_head)
        return bag_filtered