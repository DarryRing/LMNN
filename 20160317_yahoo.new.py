# -*- coding: utf-8 -*-

import os
import time
import pandas
from sklearn import neighbors
import metric_learn

#Parameters
StartDate = 20151101
num_labels = 8
svd_dim = 100
k = 3
#path = 'Yahoo/' #my PC
path = '/home/data/Yahoo/' #SSH (However, no writting right.)

# write svd files
t = time.time()
for i in range(1, num_labels):
    train_label_file = '%s%s/train%d.info' %(path, StartDate, i)
    test_label_file = '%s%s/test%d.info' %(path, StartDate, i)
    train_svdvector_file = '%s.svd' %train_label_file
    test_svdvector_file = '%s.svd' %test_label_file
    if (not os.path.exists(train_svdvector_file)) or (not os.path.exists(test_svdvector_file)):
        os.system('python %ssvd.py %s %s %d' %(path, train_label_file, test_label_file, svd_dim))
    else:
        print ('Already have svd files for step %d/%d.' %(i, (num_labels - 1)))
del i
print('Writing svd files done in %fs.' % (time.time() - t) + '[' + time.strftime('%H:%M:%S') + ']')

# classify
for i in range(1, num_labels):
    train_vectors = pandas.read_table('%s%s/train%d.info.svd' %(path, StartDate, i), sep=' ', header=None).as_matrix()
    train_labels = pandas.read_table('%s%s/train%d.info' %(path, StartDate, i), sep='\t', header=None, usecols=[1]).as_matrix().flatten()
    test_vectors = pandas.read_table('%s%s/test%d.info.svd' %(path, StartDate, i), sep=' ', header=None).as_matrix()
    test_labels = pandas.read_table('%s%s/test%d.info' %(path, StartDate, i), sep='\t', header=None, usecols=[1]).as_matrix().flatten()

    #without lmnn
    t = time.time()
    print('Classyfing in step %d/%d without lmnn...' %(i, (num_labels - 1)))
    classifier = neighbors.KNeighborsClassifier(n_neighbors=k)
    classifier.fit(train_vectors, train_labels)
    accuracy = classifier.score(test_vectors, test_labels)
    print('\taccuracy: %f.' %(accuracy))
    print("\tdone in %fs." % (time.time() - t) + '[' + time.strftime('%H:%M:%S') + ']')
    
    #with lmnn (too slow to run)
    t = time.time()
    print('Classyfing in step %d/%d with lmnn...' %(i, (num_labels - 1)))
    lmnn = metric_learn.LMNN(k=k).fit(train_vectors, train_labels)
    train_vectors = lmnn.transform(train_vectors)
    test_vectors = lmnn.transform(test_vectors)
    classifier.fit(train_vectors, train_labels)
    accuracy = classifier.score(test_vectors, test_labels)
    print('\taccuracy: %f.' %(accuracy))
    print("\tdone in %fs." % (time.time() - t) + '[' + time.strftime('%H:%M:%S') + ']')
del i