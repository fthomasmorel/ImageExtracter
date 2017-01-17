from optparse import OptionParser
from shutil import copyfile
import random
import os
import sys
import operator
import copy

def chunk(seq, num):
  avg = len(seq) / float(num)
  out = []
  last = 0.0
  while last < len(seq):
    out.append(seq[int(last):int(last + avg)])
    last += avg
  return out

def diff(l1, l2):
    result = copy.deepcopy(l1)
    for el in l2:
        if el in result:
            result.remove(el)
    return result

if __name__ == '__main__':
    random.seed(12345)
    parser = OptionParser(usage="usage: generator --images /path/to/images",
                          version="%prog 1.0")
    parser.add_option("-i", "--images",
                      dest="image_directory",
                      default="",
                      type="string",
                      help="Path to images to classify")
    (options, args) = parser.parse_args()
    image_directory = options.image_directory

    print "Generating Training and Testing sets..."

    for dirname, _, filenames in os.walk(image_directory):
        numberFiles = len(filenames)
        files = random.sample(range(0, numberFiles), int(numberFiles))
        files_index = chunk(files, 3)

        train_path = './training'
        test_path = './test'
        if not os.path.exists(train_path):
            os.makedirs(train_path)
        if not os.path.exists(test_path):
            os.makedirs(test_path)

        for i in range(0,3):
            testing_files = files_index[i]
            training_files = diff(files,files_index[i])
            train_path = os.path.join('./training/', str(i))
            test_path = os.path.join('./test/', str(i))
            if not os.path.exists(train_path):
                os.makedirs(train_path)
            if not os.path.exists(test_path):
                os.makedirs(test_path)

            for filename in testing_files:
                filename = filenames[filename]
                src = os.path.join(dirname, filename)
                dst = os.path.join(test_path, filename)
                copyfile(src, dst)
            for filename in training_files:
                filename = filenames[filename]
                src = os.path.join(dirname, filename)
                dst = os.path.join(train_path, filename)
                copyfile(src, dst)

        print "Generated Training and Testing sets!"
        exit()
