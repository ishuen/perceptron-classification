#!/usr/bin/env python
import sys
import porter
import string
import re
import numpy
import pickle
import tc_test
import timeit

def init_data(word_list, words_count, classes, txt_idx, texts):
    p = porter.PorterStemmer()
    infile = open(classes[0, txt_idx], 'r', encoding = "ISO-8859-1")
    texts[txt_idx] = []
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
                        add_wlist(word, words, words_count, txt_idx, classes)
                        texts[txt_idx].append(word)
                        word = ''
                    else:
                        word = ''
                        continue
    infile.close()
    
def add_wlist(word, word_list, word_count, txt_idx, classes):
    txt_num = len(classes[0])
    if word not in word_list:
        word_list[word] = numpy.zeros([text_num, 1])
        word_count[word] = 1
    else:
        word_count[word] += 1
    word_list[word][txt_idx, 0] = 1

def txt_class(classes):
    txt_num = len(classes[0])
    table = numpy.zeros((1, txt_num))
    class_list = []
    prev_c = classes[1][0]
    c_idx = 0
    for i in range(0, txt_num):
        if prev_c not in classes[1, i]:
            class_list = numpy.append(class_list, prev_c)
            table = numpy.append(table, numpy.zeros((1, txt_num)), axis=0)
            prev_c = classes[1][i]
            c_idx += 1
        table[c_idx, i] = 1
    class_list = numpy.append(class_list, prev_c)
    class_list = list(class_list)
    return class_list, table.transpose()

def chi2(word_doc, doc_class):
    doc_num = len(word_doc)
    class_num = len(doc_class[0])
    n11 = numpy.dot(word_doc.transpose(), doc_class)
    n01 = doc_class.sum(axis = 0).reshape((1, -1))
    n01 = n01 - n11
    n10 = numpy.full((1, class_num), word_doc.sum(axis = 0)) - n11
    n00 = numpy.full((1, class_num), doc_num) - n11 - n01 - n10
    numpy.seterr(divide='ignore', invalid='ignore')
    chi2_v = (n11 + n10 + n01 + n00) * numpy.power((n11 * n00 - n10 * n01), 2) /\
     ((n11 + n01) * (n11 + n10) * (n10 + n00) * (n01 + n00))
    chi2_v[ ~ numpy.isfinite( chi2_v )] = 0
    return chi2_v

def cross_init(classes):
    length = len(classes[0])
    grouping = numpy.zeros((length, 5), dtype=numpy.int)
    k = 0
    for i in range(0, length):
        grouping[i, k] = 1
        k = k + 1 if k != 4 else 0
    return grouping

def cross_vali(words, word_count, group, doc_class, class_num, existed_class, texts):
    word_table = [[], [], [], [], []]
    chi2_table =[numpy.empty([0, class_num])] * 5
    thredsholds = [15, 10, 5] # descending order
    accuracies = [0, 0, 0]
    for n in range(0, len(thredsholds)):
        for key, value in words_count.items():
            count = words_count[key]
            if n == 0 and count > thredsholds[n]:
                repeated = numpy.tile(words[key], (1, 5))
                masked = numpy.ma.array(repeated, mask= group, fill_value= 0)
                temp = chi2(masked.filled(), doc_class)
                for m in range(0, 5):
                    word_table[m] = numpy.append(word_table[m], key)
                    chi2_table[m] = numpy.append(chi2_table[m], [temp[m]], axis=0)
            elif n > 0 and count > thredsholds[n] and count <= thredsholds[n-1]:
                repeated = numpy.tile(words[key], (1, 5))
                masked = numpy.ma.array(repeated, mask= group, fill_value= 0)
                temp = chi2(masked.filled(), doc_class)
                for m in range(0, 5):
                    word_table[m] = numpy.append(word_table[m], key)
                    chi2_table[m] = numpy.append(chi2_table[m], [temp[m]], axis=0)
        print(n, thredsholds[n], len(word_table[3]), len(chi2_table[3]))
        for i in range(0, 5):
            chi2_temp = (chi2_table[i] != chi2_table[i].max(axis=1)[:,None]).astype(int)
            t = numpy.ma.array(chi2_table[i], mask=chi2_temp, fill_value= 0)
            chi2_table[i] = t.filled()
        # print(chi2_table[i])
        accuracies[n] = dev_test(word_table, chi2_table, classes, group, existed_class, texts)
    print(accuracies)
    max_idx = numpy.argmax(accuracies)
    # print(max_idx)
    return thredsholds[max_idx]

def dev_test(word_table, chi2_table, classes, group, existed_class, texts):
    row_num = numpy.sum(group, axis = 0)
    docs = [numpy.zeros((row_num[0], len(word_table[0]))), numpy.zeros((row_num[1],\
     len(word_table[1]))), numpy.zeros((row_num[2], len(word_table[2]))), numpy.zeros((row_num[3],\
     len(word_table[3]))), numpy.zeros((row_num[4], len(word_table[4])))]
    txt_num = len(classes[0])
    for i in range(0, txt_num):
        dev_init_test(word_table, i, docs, group, texts)
    results = []
    for i in range(0, 5):
        results = numpy.append(results, tc_test.select_class(docs[i], chi2_table[i], existed_class))
    # print (results)
    order = numpy.argwhere(group.transpose() == 1)
    acc = numpy.zeros(5)
    for i in range(0, txt_num):
        idx = order[i, 1]
        gidx = order[i, 0]
        if classes[1, idx] == results[i]:
            acc[gidx] += 1
    acc = acc/ row_num
    # print (acc)
    return numpy.average(acc)

def dev_init_test(word_list, i, docs, group, texts):
    group_idx = int(numpy.argwhere(group[i] == 1))
    k = -1
    for j in range(0, i+1):
        if group[j, group_idx] == 1:
            k += 1
    for word in texts[i]:
        if word in word_list[group_idx]:
            idx = int(numpy.argwhere(word_list[group_idx] == word))
            docs[group_idx][k, idx] += 1
    # print (docs[group_idx])

if __name__ == '__main__':
    start = timeit.default_timer()
    stopword_list = open(sys.argv[1], 'r')
    global stopword
    stopword = stopword_list.read()
    stopword_list.close()
    stopword = stopword.split()
    class_list = open(sys.argv[2], 'r')
    classes = class_list.read()
    class_list.close()
    classes = numpy.array(classes.split())
    classes = numpy.reshape(classes, (-1, 2))
    words = {}
    words_count = {}
    text_num = len(classes)
    classes = classes.transpose()
    class_list, doc_class = txt_class(classes)
    class_num = len(class_list)
    group = cross_init(classes)
    texts = [None] * text_num
    for i in range(0, text_num):
        init_data(words, words_count, classes, i, texts)
    word_table = []
    chi2_table = numpy.empty([0, class_num])
    thredshold = cross_vali(words, words_count, group, doc_class, class_num, class_list, texts)
    for key, value in words_count.items():
        if words_count[key] < thredshold:
            del words[key]
        else:
            temp = chi2(words[key], doc_class)
            word_table = numpy.append(word_table, key)
            chi2_table = numpy.append(chi2_table, temp, axis=0)
    with open(sys.argv[3], 'wb') as model:
        pickle.dump(list(class_list), model)
        pickle.dump(list(word_table), model)
        pickle.dump(chi2_table, model)
        model.close()
    print (timeit.default_timer() - start, 'sec')