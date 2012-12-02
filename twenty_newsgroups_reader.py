import string
import sys
import nltk.tokenize
import re

# Returns lists of word tokens free of punctuation marks
def load_text(in_file_name):
    in_file = open(in_file_name)
    from_line = in_file.readline().strip()
    subject_line = in_file.readline().strip()
    organization_line = in_file.readline().strip()
    lines_number_line = in_file.readline().strip()

    text = []
    for line in in_file:
        # for various email quotations
        line = line.lstrip(string.punctuation)
        text.append(line.strip())
    # tokenized punctuation-free sentences
    result = []
    for sentence in nltk.sent_tokenize(' '.join(text)):
        result.append([word.lower() for word in nltk.word_tokenize(sentence) \
                       if not re.match('^[^\w]+$', word)])
    return result



if __name__ == '__main__':
    if len(sys.argv) < 2:
        exit('usage: 20news_reader.py <text file>')
    print load_file(sys.argv[1])
