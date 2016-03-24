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
    estimate_all_labels = CLASSIFIER.predict(test_vectors)
    n_true = n_estimated = n_right = 0.0
    for i in range(len(estimate_all_labels)):
        if TEST_LABELS[i] == cur_label:#正确的Label跟判没判断过无关，所以不像下面一样忽略掉
            n_true += 1
        if not ignore_indexes[i]:#忽略已经判断好了的test数据
            if estimate_all_labels[i] == cur_label:
                n_estimated += 1
                estimate_labels[i] = cur_label
                ignore_indexes[i] = True
            if estimate_all_labels[i] == cur_label and TEST_LABELS[i] == cur_label:
                n_right += 1
    #compute_presion_recall_F
    if n_right != 0: 
        presion = n_right/n_estimated
        recall = n_right/n_true
        F = 2*presion*recall/(presion + recall)
    else: 
        presion = recall = F = 0.0
    return (presion, recall, F)
    
def compute_accuracy(estimate_labels):
    n_right = 0.0
    for i in range(len(estimate_labels)):
        if estimate_labels[i] == TEST_LABELS[i]:
            n_right += 1
    return n_right/len(estimate_labels)

#Parameters
START_DATE = 20151101
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


#test不需要每次都重新读取
TEST_LABELS = pandas.read_table('%s%s/test1.info' %(PATH, START_DATE), sep='\t', header=None, usecols=[1]).as_matrix().flatten()
test_vectors = pandas.read_table('%s%s/test1.info.svd' %(PATH, START_DATE), sep=' ', header=None).as_matrix()#因为用lmnn时会变，所以与TEST_LABELS不同，不用作全局变量
CLASSIFIER = neighbors.KNeighborsClassifier(n_neighbors=N_NEIGHBORS)

#记录结果
presion_withoutlmnn = {}
recall_withoutlmnn = {}
F_withoutlmnn = {}
#陆续加
estimate_labels_withoutlmnn = ['' for i in range(len(TEST_LABELS))]
ignore_indexes_withoutlmnn = [False for i in range(len(TEST_LABELS))]

#记录结果
presion_withlmnn = {}
recall_withlmnn = {}
F_withlmnn = {}
#陆续加
estimate_labels_withlmnn = ['' for i in range(len(TEST_LABELS))]
ignore_indexes_withlmnn = [False for i in range(len(TEST_LABELS))]

# classify
for i in range(len(ORDER_LABELS) - 1):
    cur_label = ORDER_LABELS[i]
    print('Classyfing label %s (step %d/%d)...' %(cur_label, i + 1, (len(ORDER_LABELS) - 1)))
    train_vectors = pandas.read_table('%s%s/train%d.info.svd' %(PATH, START_DATE, i + 1), sep=' ', header=None).as_matrix()
    train_labels = pandas.read_table('%s%s/train%d.info' %(PATH, START_DATE, i + 1), sep='\t', header=None, usecols=[1]).as_matrix().flatten()

    #without lmnn
    t = time.time()
    print('SVD...')
    (presion_withoutlmnn[cur_label], recall_withoutlmnn[cur_label], F_withoutlmnn[cur_label]) = \
        classification(cur_label, train_vectors, train_labels, test_vectors, estimate_labels_withoutlmnn, ignore_indexes_withoutlmnn)
    print("Presion: %f, Recall: %f, F: %f." %(presion_withoutlmnn[cur_label], recall_withoutlmnn[cur_label], F_withoutlmnn[cur_label]))
    print("\tdone in %fs." % (time.time() - t) + '[' + time.strftime('%H:%M:%S') + ']')
    
    #with lmnn (too slow to run)
#    t = time.time()
#    print('SVD + lmnn...' %(i + 1, (len(ORDER_LABELS) - 1)))
#    lmnn = metric_learn.LMNN(k=N_NEIGHBORS).fit(train_vectors, train_labels)
#    (presion_withlmnn[cur_label], recall_withlmnn[cur_label], F_withlmnn[cur_label]) = \
#        classification(cur_label, lmnn.transform(train_vectors), train_labels, lmnn.transform(test_vectors), estimate_labels_withlmnn, ignore_indexes_withlmnn)
#    print("Presion: %f, Recall: %f, F: %f." %(presion_withlmnn[cur_label], recall_withlmnn[cur_label], F_withlmnn[cur_label]))
#    print("\tdone in %fs." % (time.time() - t) + '[' + time.strftime('%H:%M:%S') + ']')
#    print()
del i

accuracy_withoutlmnn = compute_accuracy(estimate_labels_withoutlmnn)
accuracy_withlmnn = compute_accuracy(estimate_labels_withlmnn)
print('Accuracy without lmnn: %f.' %accuracy_withoutlmnn)
print('Accuracy with lmnn: %f.' %accuracy_withlmnn)