import text_reading

def read_file_into_map(in_file_name):
    sentences = text_reading.twenty_newsgroups.load_text(in_file_name)
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
