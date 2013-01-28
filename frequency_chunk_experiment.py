import sys
import nltk.corpus

import classify
import frequency_chunking
import twenty_newsgroups_reader
import text_reading
import text_reading.twenty_newsgroups
import text_reading.ruscorpora
import dataset_loading

REMOVE_STOPWORDS = True
def tail_cutting_experiment(in_train_set, in_test_set):
    for chunks_number in xrange(4):
        freq_chunk_filter = frequency_chunking.FrequencyChunkFilterWrapper(in_chunks_number = 10, \
                                                                    in_cut_tail = chunks_number)
        quality = classify.classify_texts(in_train_set, in_test_set, freq_chunk_filter)
        print '%d tail chunks cut: classification quality = %f' % (chunks_number, quality)

def head_cutting_experiment(in_train_set, in_test_set):
    for chunks_number in xrange(4):
        freq_chunk_filter = frequency_chunking.FrequencyChunkFilterWrapper(in_chunks_number = 10,
                                                                    in_cut_head = chunks_number)
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
        freq_chunk_filter = frequency_chunking.FrequencyChunkFilterWrapper(in_cut_head = cut_head,
                                                                    in_cut_tail = cut_tail)
        quality = classify.classify_texts(in_train_set, in_test_set, freq_chunk_filter)
        print '%d -- %d chunks used: classification quality = %f' % \
              (begin_chunk, begin_chunk + window_size - 1, quality)

if __name__ == '__main__':
    if len(sys.argv) < 4:
        exit('Usage: frequency_chunk_experiment.py \
             <head|tail|window> <training data folder> <testing data root> \
             [dataset=ruscorpora|20newsgroups]')

    dataset_type = None
    if len(sys.argv) == 5:
        dataset_type = sys.argv[4]
    sentences_extractor = None
    stop_list = []
    if dataset_type == '20newsgroups':
        sentences_extractor = getattr(text_reading.twenty_newsgroups, 'load_text_raw')
        if REMOVE_STOPWORDS:
            stop_list = [word for word in nltk.corpus.stopwords.words('english')]
    elif dataset_type == 'ruscorpora':
        sentences_extractor = getattr(text_reading.ruscorpora, 'get_text_raw')
        if REMOVE_STOPWORDS:
            stop_list = [word.decode('utf-8') for word in nltk.corpus.stopwords.words('russian')]
    train_dataset = dataset_loading.DatasetLoader(sys.argv[2],
                                                  sentences_extractor,
                                                  in_stop_list = stop_list)
    test_dataset = dataset_loading.DatasetLoader(sys.argv[3],
                                                 sentences_extractor,
                                                 in_stop_list = stop_list)

    if sys.argv[1] == 'head':
        head_cutting_experiment(train_dataset, test_dataset)
    elif sys.argv[1] == 'tail':
        tail_cutting_experiment(train_dataset, test_dataset)
    elif sys.argv[1] == 'window':
        chunk_window_experiment(train_dataset, test_dataset)