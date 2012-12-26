import sys

import classify
import frequency_chunking
import twenty_newsgroups_reader

def tail_cutting_experiment(in_train_set, in_test_set):
    for chunks_number in xrange(15):
        freq_chunk_filter = frequency_chunking.FrequencyChunkFilter(in_cut_tail = chunks_number)
        quality = classify.classify_texts(in_train_set, in_test_set, freq_chunk_filter)
        print '%d tail chunks cut: classification quality = %f' % (chunks_number, quality)

def head_cutting_experiment(in_train_set, in_test_set):
    for chunks_number in xrange(15):
        freq_chunk_filter = frequency_chunking.FrequencyChunkFilter(in_cut_head = chunks_number)
        quality = classify.classify_texts(in_train_set, in_test_set, freq_chunk_filter)
        print '%d head chunks cut: classification quality = %f' % (chunks_number, quality)

def get_filtered_distribution_test(cut_head = 0, cut_tail = 0):
        result = {}
        if cut_head == 0:
            cut_head = None
        else:
            cut_head = -cut_head
        # distribution tail == low frequencies == partitioning begin
        # distribution tail == high frequencies == partitioning end
        (cut_left, cut_right) = (cut_tail, cut_head)
        return (str(cut_left), str(cut_right))

def chunk_window_experiment(in_train_set, in_test_set):
    window_size = 5
    for begin_chunk in xrange(frequency_chunking.DEFAULT_CHUNKS_NUMBER - window_size + 1):
        cut_head = begin_chunk
        cut_tail = frequency_chunking.DEFAULT_CHUNKS_NUMBER - begin_chunk - window_size
        #(cut_left, cut_right) = get_filtered_distribution_test(cut_head, cut_tail)
        # print "%d,%d --> [%s:%s]" % (cut_head, cut_tail, cut_left, cut_right)
        freq_chunk_filter = frequency_chunking.FrequencyChunkFilter(in_cut_head = cut_head,
                                                                    in_cut_tail = cut_tail)
        quality = classify.classify_texts(in_train_set, in_test_set, freq_chunk_filter)
        print '%d -- %d chunks used: classification quality = %f' % \
              (begin_chunk, begin_chunk + window_size - 1, quality)

if __name__ == '__main__':
    if len(sys.argv) < 4:
        exit('Usage: frequency_chunk_experiment.py \
             <head|tail|window> <training data folder> <testing data root>')
    train_set = twenty_newsgroups_reader.DatasetLoader(sys.argv[2])
    test_set = twenty_newsgroups_reader.DatasetLoader(sys.argv[3])
    if sys.argv[1] == 'head':
        head_cutting_experiment(train_set, test_set)
    elif sys.argv[1] == 'tail':
        tail_cutting_experiment(train_set, test_set)
    elif sys.argv[1] == 'window':
        chunk_window_experiment(train_set, test_set)