# -*- coding: utf-8 -*-
#只记录要忽略的test数据的index，并不对test_vector和test_label进行修改
#without_lmnn: only svd; with_lmnn: svd + lmnn

import os
import time
import pandas
from sklearn import neighbors
import metric_learn
#import copy
#import numpy as np

def classification(cur_label, train_vectors, train_labels, test_vectors, estimate_labels, ignore_indexes):
    CLASSIFIER.fit(train_vectors, train_labels)
    estimate_labels_cur = CLASSIFIER.predict(test_vectors)
    for i in range(len(estimate_labels_cur)):
        if not ignore_indexes[i]:#忽略已经判断好了的test数据
            if estimate_labels_cur[i] == cur_label:
                estimate_labels[i] = cur_label
                ignore_indexes[i] = True
    
def compute_accuracy(estimate_labels):
    #初始化
    n_right = {}
    n_estimated = {}
    n_true = {}
    for label in ORDER_LABELS:
        n_right[label] = 0.0
        n_estimated[label] = 0.0
        n_true[label] = 0.0
    del label
    #统计    
    n_right_all = 0.0
    for i in range(len(estimate_labels)):
        n_estimated[estimate_labels[i]] += 1
        n_true[TEST_LABELS[i]] += 1
        if estimate_labels[i] == TEST_LABELS[i]:
            n_right[estimate_labels[i]] += 1
            n_right_all += 1
    del i    
     #计算       
    presion = {}
    recall = {}
    F = {}      
    for label in ORDER_LABELS:
        if n_right[label] != 0: 
            presion[label] = n_right[label]/n_estimated[label]
            recall[label] = n_right[label]/n_true[label]
            F[label] = 2*presion[label]*recall[label]/(presion[label] + recall[label])
        else: 
            presion[label] = recall[label] = F[label] = 0.0
    del label
    return (presion, recall, F, n_right_all/len(estimate_labels))

def printout(method, presion, recall, F, accuracy):
    print('Accuracy %s: %f.' %(method, accuracy))
    for label in ORDER_LABELS:
        print("Presion: %f, Recall: %f, F: %f." %(presion[label], recall[label], F[label]))
    del label
    
def printfile(method, labels):
    file = open('labels_%s.csv' %(method), 'w')
    file.write("index,label\n")
    for i in range(len(labels)):
        file.write('%d,%s\n' %(i, labels[i]))
    del i
    file.close()

def supplementLastLabel(estimated_labels):
    for i in range(len(estimated_labels)):
        if estimated_labels[i] == '':
            estimated_labels[i] = ORDER_LABELS[-1]
    del i

#Parameters
START_DATE = 20151101
NUM_DATA = 16345
SVD_DIM = 100
N_NEIGHBORS = 3
ORDER_LABELS = ['スポーツ','エンタメ','国際','国内','経済','地域','ライフ','IT・科学']
PATH = 'Yahoo/' #my PC
#PATH = '/home/data/Yahoo/' #SSH (However, no writting right.)

# write svd files
t = time.time()
for i in range(len(ORDER_LABELS) - 1):
    train_label_file = '%s%s/train%d.info' %(PATH, START_DATE, i + 1)
    test_label_file = '%s%s/test%d.info' %(PATH, START_DATE, i + 1)
    train_svdvector_file = '%s.svd' %train_label_file
    test_svdvector_file = '%s.svd' %test_label_file
    if (not os.path.exists(train_svdvector_file)) or (not os.path.exists(test_svdvector_file)):
        os.system('python %ssvd.py %s %s %d' %(PATH, train_label_file, test_label_file, SVD_DIM)) #have no right to write files when using SSH
    else:
        print ('Already have svd files for step %d/%d.' %(i + 1, (len(ORDER_LABELS) - 1)))
del i, train_label_file, test_label_file, train_svdvector_file, test_svdvector_file
print('Writing svd files done in %fs.' % (time.time() - t) + '[' + time.strftime('%H:%M:%S') + ']')


#test的LABEL不需要每次都重新读取
TEST_LABELS = pandas.read_table('%s%s/test1.info' %(PATH, START_DATE), sep='\t', header=None, usecols=[1]).as_matrix().flatten()
CLASSIFIER = neighbors.KNeighborsClassifier(n_neighbors=N_NEIGHBORS)

#记录结果
presion_withoutlmnn = {}
recall_withoutlmnn = {}
F_withoutlmnn = {}
#陆续加
estimate_labels_withoutlmnn = ['' for i in range(NUM_DATA)]
ignore_indexes_withoutlmnn = [False for i in range(NUM_DATA)]

#记录结果
presion_withlmnn = {}
recall_withlmnn = {}
F_withlmnn = {}
#陆续加
estimate_labels_withlmnn = ['' for i in range(NUM_DATA)]
ignore_indexes_withlmnn = [False for i in range(NUM_DATA)]

# classify
for i in range(len(ORDER_LABELS) - 1):
    cur_label = ORDER_LABELS[i]
    print('Classyfing label %s (step %d/%d)...' %(cur_label, i + 1, (len(ORDER_LABELS) - 1)))
    train_vectors = pandas.read_table('%s%s/train%d.info.svd' %(PATH, START_DATE, i + 1), sep=' ', header=None).as_matrix()
    train_labels = pandas.read_table('%s%s/train%d.info' %(PATH, START_DATE, i + 1), sep='\t', header=None, usecols=[1]).as_matrix().flatten()
    #test的VECTOR需要每次都重新读取
    test_vectors = pandas.read_table('%s%s/test%d.info.svd' %(PATH, START_DATE, i + 1), sep=' ', header=None).as_matrix()
    #without lmnn
    t = time.time()
    print('SVD...')
    classification(cur_label, train_vectors, train_labels, test_vectors, estimate_labels_withoutlmnn, ignore_indexes_withoutlmnn)
    print("\tdone in %fs." % (time.time() - t) + '[' + time.strftime('%H:%M:%S') + ']')
    
    #with lmnn (too slow to run)
#    t = time.time()
#    print('SVD + lmnn...')
#    lmnn = metric_learn.LMNN(k=N_NEIGHBORS).fit(train_vectors, train_labels)
#    classification(cur_label, lmnn.transform(train_vectors), train_labels, lmnn.transform(test_vectors), estimate_labels_withlmnn, ignore_indexes_withlmnn)
#    print("\tdone in %fs." % (time.time() - t) + '[' + time.strftime('%H:%M:%S') + ']')
    
    print()
del i

#supplement the last label
supplementLastLabel(estimate_labels_withoutlmnn)
supplementLastLabel(estimate_labels_withlmnn)

printfile('estimated_withoutlmnn', estimate_labels_withoutlmnn)
printfile('estimated_withlmnn', estimate_labels_withlmnn)
printfile('true', TEST_LABELS)
(presion_withoutlmnn, recall_withoutlmnn, F_withoutlmnn, accuracy_withoutlmnn) = compute_accuracy(estimate_labels_withoutlmnn)
(presion_withlmnn, recall_withlmnn, F_withlmnn, accuracy_withlmnn) = compute_accuracy(estimate_labels_withlmnn)
printout('without lmnn', presion_withoutlmnn, recall_withoutlmnn, F_withoutlmnn, accuracy_withoutlmnn)
printout('with lmnn', presion_withlmnn, recall_withlmnn, F_withlmnn, accuracy_withlmnn)