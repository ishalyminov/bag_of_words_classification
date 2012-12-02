import twenty_newsgroups_reader

class TfIdfBagOfWordsBuilder(object):
    def __init__(self):
        self.terms_dictionary = collections.default_dict(lambda: 0)
        self.document_counter = 0
        self.inverted_index = collections.default_dict(lambda: set([]))
        self.weights = {}

    def add_document(self, in_sentences):
        for sentence in in_sentences:
            for word in sentence:
                self.terms_dictionary[word] += 1
                self.inverted_index[word].add(self.document_counter)
        self.document_counter += 1
        self.calculate_weights()

    def calculate_weights(self):
        for (term, count) in self.terms_dictionary.iteritems():
            self.weights[term] = float(count) / float(len(self.inverted_index(term)))

    def get_weight(self, in_word):
        return self.weights(in_word) if in_word in self.weights else 0.;

def read_file_into_map(in_file_name):
    sentences = twenty_newsgroups_reader.load_text(in_file_name)
    return sentences_to_bag_of_words(sentences)

def sentences_to_bag_of_words(in_sentences):
    bag_of_words = {}
    for sentence in in_sentences:
        for word in sentence:
            if word in bag_of_words:
                bag_of_words[word] += 1
            else:
                bag_of_words[word] = 1
    return bag_of_words
