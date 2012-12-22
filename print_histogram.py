import sys
import twenty_newsgroups_reader
import collections
import operator
import os
import errno

# how many elements of a histogram to print (None means entire histogram)
TOP_ELEMENTS_NUMBER = None

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def process_dirs_synchronously(in_src_root, in_dst_root):
    mkdir_p(in_dst_root)
    for root, folders, files in os.walk(in_src_root, followlinks = True):
        for file_name in files:
            src_file_name = os.path.join(root, file_name)
            dst_file_name = os.path.join(in_dst_root, src_file_name[len(in_src_root):].lstrip('/'))
            print '%s --> %s' % (src_file_name, dst_file_name)
            process_file(src_file_name, dst_file_name)
        for folder in folders:
            src_folder_name = os.path.join(root, folder)
            dst_folder_name = os.path.join(in_dst_root, src_folder_name[len(in_src_root):].lstrip('/'))
            print '%s --> %s' % (src_folder_name, dst_folder_name)
            mkdir_p(dst_folder_name)




def process_file(in_file_name, out_file_name):
    bag = collections.defaultdict(lambda: 0)
    sentences = twenty_newsgroups_reader.load_text(in_file_name)
    for sentence in sentences:
        for word in sentence:
            bag[word.lower()] += 1

    out_file = open(out_file_name, 'w')
    for (key, value) in sorted(bag.items(), key=operator.itemgetter(1), reverse=True)[:TOP_ELEMENTS_NUMBER]:
        print >>out_file, '%s\t%d' % (key, value)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        exit('Usage: print_histogram.py <source folder> <destination folder>')
    process_dirs_synchronously(os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]))