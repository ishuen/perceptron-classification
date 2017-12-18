import numpy
import pickle
import sys
import porter
import string
import timeit

def init_test(doc, word_list, i, docs):
    p = porter.PorterStemmer()
    infile = open(doc, 'r', encoding = "ISO-8859-1")
    while 1:
        word = ''
        line = infile.readline()
        if line == '':
            break
        elif line == '\n':
            continue
        for c in line:
            if c.isalpha():
                word += c.lower()
            else:
                if c.isdigit():
                    continue
                elif c in string.punctuation:
                    continue
                elif word:
                    if not word in stopword:
                        word = p.stem(word, 0,len(word)-1)
                        if word in word_list:
                            idx = word_list.index(word)
                            docs[i, idx] += 1
                        word = ''
                    else:
                        word = ''
                        continue
    infile.close()

def select_class(doc, chi2_table, existed_class):
    sum_product = numpy.dot(doc, chi2_table)
    temp = numpy.argmax(sum_product, axis=1)
    idxs = [None] * len(temp)
    for i in range(0, len(temp)):
        idxs[i] = existed_class[temp[i]]
    return idxs

if __name__ == '__main__':
    start = timeit.default_timer()
    model = open(sys.argv[2], 'rb')
    existed_class = pickle.load(model)
    word_list = pickle.load(model)
    chi2_table = pickle.load(model)
    # print (chi2_table)
    stopword_list = open(sys.argv[1], 'r')
    global stopword
    stopword = stopword_list.read()
    stopword_list.close()
    stopword = stopword.split()
    class_list = open(sys.argv[3], 'r')
    classes = class_list.read()
    class_list.close()
    classes = numpy.array(classes.split())
    txt_num = len(classes)
    classes = numpy.reshape(classes, (-1, txt_num))
    classes = numpy.append(classes, numpy.empty((1, txt_num)), axis=0)
    docs = numpy.zeros((txt_num, len(word_list)))
    for i in range(0, txt_num):
        init_test(classes[0, i], word_list, i, docs)
    classes[1] = select_class(docs, chi2_table, existed_class)
    classes = classes.transpose()
    output = open(sys.argv[4], 'wb')
    numpy.savetxt(output, classes, fmt='%s')
    print (timeit.default_timer() - start, 'sec')