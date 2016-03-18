#encoding='shift_jis', UTF-8
#0, 0.0
#\\,/
#together 4000>= : MemoryError

#steps: run -> classify_and_test -> statistic -> output

from sklearn import neighbors
import metric_learn #Download: https://pypi.python.org/pypi/metric-learn/
import time
import pandas
import numpy
import copy
import os
import shutil

#[300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
candidate_n_train_lmnn = [300]
candidate_n_neighbors = [3,5]
strategy = 'together' #'together' or 'separately'
daytype = 'alldays' #'everyday' or 'alldays'
dataset = 'svd' #'topic', 'topic_small' or 'svd'
n_max_iter = 1000#default:1000, for test
is_run_without_lmnn = False

path = 'Results/'
if not os.path.exists(path): os.mkdir(path)
path_strategy = path + strategy + '/'
del path
if not os.path.exists(path_strategy): os.mkdir(path_strategy)
path_daytype = path_strategy + daytype + '/'
if not os.path.exists(path_daytype): os.mkdir(path_daytype)
try: shutil.rmtree(path_daytype) 
except: None
try: os.mkdir(path_daytype)
except: None

print('Importing data...')
if dataset == 'topic' or 'svd':
    if dataset == 'topic':
        train_vectors = pandas.read_table('Yahoo/train.topic', sep=' ', header=None).as_matrix()
        test_vectors = pandas.read_table('Yahoo/test.topic', sep=' ', header=None).as_matrix()
    elif dataset == 'svd':
        train_vectors = pandas.read_table('Yahoo/train.svd', sep=' ', header=None).as_matrix()
        test_vectors = pandas.read_table('Yahoo/test.svd', sep=' ', header=None).as_matrix()
    all_train_labels = pandas.read_table('Yahoo/train.info', sep='\t', header=None, usecols=[4]).as_matrix().flatten()
    all_test_labels = pandas.read_table('Yahoo/test.info', sep='\t', header=None, usecols=[4]).as_matrix().flatten()
    test_dates = pandas.read_table('Yahoo/test.info', sep='\t', header=None, usecols=[0], dtype=str).as_matrix().flatten()
elif dataset == 'topic_small':
    train_vectors = pandas.read_table('Yahoo/train.topic.small', sep=' ', header=None).as_matrix()
    test_vectors = pandas.read_table('Yahoo/test.topic.small', sep=' ', header=None).as_matrix()
    all_train_labels = pandas.read_table('Yahoo/train.info.small', sep='\t', header=None, usecols=[4]).as_matrix().flatten()
    all_test_labels = pandas.read_table('Yahoo/test.info.small', sep='\t', header=None, usecols=[4]).as_matrix().flatten()
    test_dates = pandas.read_table('Yahoo/test.info.small', sep='\t', header=None, usecols=[0], dtype=str).as_matrix().flatten()
else: print('Wrong dataset (topic, small_topic, or svd).')
labels = numpy.unique(all_test_labels)

#train, test, output label files; alldays or everyday
def classify_and_test(filename, n_neighbors, train_vectors, train_labels, test_vectors, test_labels):
    #train
    print('Classifying ' + daytype + ' ' + filename + '... [' + time.strftime('%H:%M:%S') + ']')
    classifier = neighbors.KNeighborsClassifier(n_neighbors)
    classifier.fit(train_vectors, train_labels)
    #test
    dic_day_test_vectors = {}
    dic_day_test_labels = {}
    if daytype == 'alldays':
        dic_day_test_vectors['alldays'] = test_vectors
        dic_day_test_labels['alldays'] = test_labels
    elif daytype == 'everyday':
        for i in range(len(test_dates)):
            day = test_dates[i]
            if day in dic_day_test_vectors.keys():
                dic_day_test_vectors[day].append(test_vectors[i])
                dic_day_test_labels[day].append(test_labels[i])
            else:
                dic_day_test_vectors[day]=[test_vectors[i]]
                dic_day_test_labels[day]=[test_labels[i]]
        del i
    else: print('daytype input error (everyday or alldays).')
    
    dic_day_accuracy = {}
    dic_day_estimated_label = {}
    for day in sorted(dic_day_test_labels.keys()):
        dic_day_accuracy[day] = classifier.score(dic_day_test_vectors[day], dic_day_test_labels[day])
        dic_day_estimated_label[day] = classifier.predict(dic_day_test_vectors[day])
        indicators = numpy.ones(len(dic_day_test_labels[day]), numpy.int8)
        for i in range(len(dic_day_test_labels[day])):
            if dic_day_estimated_label[day][i] != dic_day_test_labels[day][i]: indicators[i] = 0#outputer of right or wrong
            del i
        path_name = path_daytype + 'labels/'
        if not os.path.exists(path_name): os.mkdir(path_name)
        result = pandas.DataFrame(numpy.array([dic_day_test_labels[day], dic_day_estimated_label[day], indicators]).T)
        result.to_csv(path_name + filename + '_' + day + '_label.csv', header=['True', 'Estimated', 'Y or N'], index=False, encoding='UTF-8')#, encoding='shift_jis'
        print('Accuracy of ' + day + ': %f [' %dic_day_accuracy[day] + time.strftime('%H:%M:%S') + '].')
    return(dic_day_test_labels, dic_day_estimated_label, dic_day_accuracy)

#kNN without lmnn
def run_without_lmnn(label, n_neighbors, train_vectors, train_labels, test_vectors, test_labels):
    filename = 'kNN_without_Lmnn(k=%d)_' %n_neighbors + label
    (dic_day_test_labels, dic_day_estimated_label, dic_day_accuracy) = classify_and_test(filename, n_neighbors, train_vectors, train_labels, test_vectors, test_labels)
    return (filename, dic_day_test_labels, dic_day_estimated_label, dic_day_accuracy)

#kNN with lmnn, output matric files
def run_with_lmnn(label, n_neighbors, n_train_lmnn, train_lmnn_vectors, train_lmnn_labels, train_vectors, train_labels, test_vectors, test_labels):  
    t = time.time()    
    print('Fitting ' + label + ' lmnn (n = %d) + kNN (k = %d)... [' %(n_train_lmnn, n_neighbors) + time.strftime('%H:%M:%S') + ']')
    lmnn = metric_learn.LMNN(k=n_neighbors, max_iter=n_max_iter)
    lmnn.fit(train_lmnn_vectors, train_lmnn_labels)
    path_name = path_strategy + 'metrix/'#no relationship with day
    if not os.path.exists(path_name): os.mkdir(path_name)
    pandas.DataFrame(lmnn.metric()).to_csv(path_name + 'kNN_with_Lmnn(k=%d,n=%d)_' %(n_neighbors, n_train_lmnn) + label + '_Matric.csv', header=False, index=False)
    print('\tdone in %.fs [' % (time.time() - t) + time.strftime('%H:%M:%S') + '].' )
    
    filename_partdata = 'kNN_with_Lmnn_Partdata(k=%d,n=%d)_' %(n_neighbors, n_train_lmnn) + label
    lmnn_train_vectors = lmnn.transform(train_lmnn_vectors)
    lmnn_test_vectors = lmnn.transform(test_vectors)
    (dic_day_test_labels_partdata, dic_day_estimated_label_partdata, dic_day_accuracy_partdata) \
        = classify_and_test(filename_partdata, n_neighbors, lmnn_train_vectors, train_lmnn_labels, lmnn_test_vectors, test_labels)
    
    filename_alldata = 'kNN_with_Lmnn_Alldata(k=%d,n=%d)_' %(n_neighbors, n_train_lmnn) + label
    lmnn_all_train_vectors = lmnn.transform(train_vectors)
    (dic_day_test_labels_alldata, dic_day_estimated_label_alldata, dic_day_accuracy_alldata) \
        = classify_and_test(filename_alldata, n_neighbors, lmnn_all_train_vectors, train_labels, lmnn_test_vectors, test_labels)
    return (filename_partdata, dic_day_test_labels_partdata, dic_day_estimated_label_partdata, dic_day_accuracy_partdata, \
        filename_alldata, dic_day_test_labels_alldata, dic_day_estimated_label_alldata, dic_day_accuracy_alldata)

#statistics and output statistic files
def statistics_together(filename, dic_day_true_labels, dic_day_estimated_labels, accuracy, seconds):
    print('Analyzing statistics... [' + time.strftime('%H:%M:%S') + ']')
    path_name = path_daytype + 'statistics/' 
    if not os.path.exists(path_name): os.mkdir(path_name)
    for day in sorted(dic_day_estimated_labels.keys()):    
        dic_label_true_freq = {}
        dic_label_estimated_freq = {}
        dic_label_right_freq = {}
        for label in labels:#initialize
            dic_label_true_freq[label] = dic_label_estimated_freq[label] = dic_label_right_freq[label] = 0.0
        del label
        for i in range(len(dic_day_true_labels[day])):#compute
            true_label = dic_day_true_labels[day][i]
            estimated_label = dic_day_estimated_labels[day][i]
            dic_label_true_freq[true_label] += 1    
            dic_label_estimated_freq[estimated_label] += 1
            if true_label == estimated_label: dic_label_right_freq[true_label] += 1
        del i, true_label, estimated_label
        presion = {}
        recall = {}
        F = {}
        file = open(path_name + filename + '_' + day + '_statistic.csv', 'w')#, encoding='shift_jis'
        file.write('Label,n_true,n_estimated,n_right,presion,recall,F\n')
        for label in labels:
            (presion[label], recall[label], F[label]) = compute_presion_recall_F(dic_label_true_freq[label], dic_label_estimated_freq[label], dic_label_right_freq[label])
            file.write(label + ',%d,%d,%d,%f,%f,%f\n' 
                %(dic_label_true_freq[label], dic_label_estimated_freq[label], dic_label_right_freq[label], presion[label], recall[label], F[label]))
        del label
        file.write('\nAccuracy,%f' %accuracy[day])
        file.write('\nTime,%ds' %seconds)
        file.write('\n\npresion = n_right/n_estimated\nrecall = n_right/n_true\nF = 2*presion*recall/(presion+recall)')
        file.close()

def statistic_separately(dic_day_true_states, dic_day_estimated_states, accuracy):
    print('Analyzing statistics... [' + time.strftime('%H:%M:%S') + ']')
    dic_day_statistics = {}
    for day in sorted(dic_day_estimated_states.keys()):
        n_true = n_estimated = n_right = 0.0
        for i in range(len(dic_day_true_states[day])):
            if dic_day_true_states[day][i] != 'others': n_true += 1
            if dic_day_estimated_states[day][i] != 'others': n_estimated += 1 
            if dic_day_true_states[day][i] != 'others' and dic_day_estimated_states[day][i] != 'others': n_right += 1
        (presion, recall, F) = compute_presion_recall_F(n_true, n_estimated, n_right)
        dic_day_statistics[day] = numpy.array([n_true, n_estimated, n_right, presion, recall, F, accuracy[day]])
    return dic_day_statistics

def output_separately(filename, dic2_label_day_statistics, seconds):
    path_name = path_daytype + 'statistics/' 
    if not os.path.exists(path_name): os.mkdir(path_name)
    for i in dic2_label_day_statistics.values():
        days = sorted(i.keys())
        break
    for day in days:
        file = open(path_name + filename + '_separately' + '_' + day + '_statistic.csv', 'w')#, encoding='shift_jis'
        file.write('Label,n_true,n_estimated,n_right,presion,recall,F,accuracy\n')
        for label in labels:
            file.write(label + ',%d,%d,%d,%f,%f,%f,%f\n' 
                %(dic2_label_day_statistics[label][day][0], dic2_label_day_statistics[label][day][1], dic2_label_day_statistics[label][day][2], \
                dic2_label_day_statistics[label][day][3], dic2_label_day_statistics[label][day][4], \
                dic2_label_day_statistics[label][day][5], dic2_label_day_statistics[label][day][6]))
        del label
        file.write('\nTime,%ds' %seconds)
        file.write('\npresion = n_right/n_estimated\nrecall = n_right/n_true\nF = 2*presion*recall/(presion+recall)')
        file.close()
    del day

def compute_presion_recall_F(n_true, n_estimated, n_right):
    if n_right != 0: 
        presion = n_right/n_estimated
        recall = n_right/n_true
        F = 2*presion*recall/(presion + recall)
    else: presion = recall = F = 0.0
    return (presion, recall, F)

#------main------
if strategy == 'together':
    #knn without lmnn
    if is_run_without_lmnn:
        for n_neighbors in candidate_n_neighbors:
            t = time.time()
            (filename, dic_day_test_labels, dic_day_estimated_label, dic_day_accuracy) = run_without_lmnn('together', n_neighbors, train_vectors, all_train_labels, test_vectors, all_test_labels)
            statistics_together(filename, dic_day_test_labels, dic_day_estimated_label, dic_day_accuracy, time.time()-t)
        del n_neighbors, filename, dic_day_test_labels, dic_day_estimated_label, dic_day_accuracy, t
    #knn with lmnn
    for n_train_lmnn in candidate_n_train_lmnn:
        train_lmnn_vectors = train_vectors[::len(all_train_labels)/n_train_lmnn,]
        train_lmnn_labels = all_train_labels[::len(all_train_labels)/n_train_lmnn]
        for n_neighbors in candidate_n_neighbors:
            t = time.time()
            (filename_partdata, dic_day_test_labels_partdata, dic_day_estimated_label_partdata, dic_day_accuracy_partdata, \
                filename_alldata, dic_day_test_labels_alldata, dic_day_estimated_label_alldata, dic_day_accuracy_alldata) \
                = run_with_lmnn('together', n_neighbors, n_train_lmnn, train_lmnn_vectors, train_lmnn_labels, \
                    train_vectors, all_train_labels, test_vectors, all_test_labels)
            statistics_together(filename_partdata, dic_day_test_labels_partdata, dic_day_estimated_label_partdata, dic_day_accuracy_partdata, time.time()-t)
            statistics_together(filename_alldata, dic_day_test_labels_alldata, dic_day_estimated_label_alldata, dic_day_accuracy_alldata, time.time()-t)
        del n_neighbors, filename_partdata, dic_day_test_labels_partdata, dic_day_estimated_label_partdata, dic_day_accuracy_partdata, \
            filename_alldata, dic_day_test_labels_alldata, dic_day_estimated_label_alldata, dic_day_accuracy_alldata, t
    del n_train_lmnn

elif strategy == 'separately':
    #generate data
    dic_label_states_train = {}
    dic_label_states_test = {}    
    for label in labels:
        dic_label_states_train[label] = copy.deepcopy(all_train_labels)
        for i in range(len(all_train_labels)):
            if dic_label_states_train[label][i] != label: dic_label_states_train[label][i] = 'others'
        dic_label_states_test[label] = copy.deepcopy(all_test_labels)
        for i in range(len(all_test_labels)):
            if dic_label_states_test[label][i] != label: dic_label_states_test[label][i] = 'others'
    del label, i
    #knn without lmnn
    if is_run_without_lmnn:
        for n_neighbors in candidate_n_neighbors:
            t = time.time()
            dic2_label_day_statistics = {}
            for label in labels:
                (filename, dic_day_test_states, dic_day_estimated_states, dic_day_accuracy) = run_without_lmnn(label, n_neighbors, train_vectors, dic_label_states_train[label], test_vectors, dic_label_states_test[label])
                dic2_label_day_statistics[label] = statistic_separately(dic_day_test_states, dic_day_estimated_states, dic_day_accuracy)
            del label, filename, dic_day_estimated_states, dic_day_accuracy
            output_separately('kNN_without_Lmnn(k=%d)_' %n_neighbors, dic2_label_day_statistics, time.time()-t)
        del n_neighbors, dic2_label_day_statistics, t
    #knn with lmnn
    for n_train_lmnn in candidate_n_train_lmnn:
        train_lmnn_vectors = train_vectors[::len(all_train_labels)/n_train_lmnn,]
        train_lmnn_states = {}
        for label in labels:
            train_lmnn_states[label] = dic_label_states_train[label][::len(all_train_labels)/n_train_lmnn,]
        del label
        for n_neighbors in candidate_n_neighbors:
            t = time.time()
            dic2_label_day_statistics_partdata = {}
            dic2_label_day_statistics_alldata = {}
            for label in labels:
                (filename_partdata, dic_day_test_states_partdata, dic_day_estimated_states_partdata, dic_day_accuracy_partdata, \
                filename_alldata, dic_day_test_states_alldata, dic_day_estimated_states_alldata, dic_day_accuracy_alldata) \
                    = run_with_lmnn(label, n_neighbors, n_train_lmnn, train_lmnn_vectors, train_lmnn_states[label], train_vectors, dic_label_states_train[label], test_vectors, dic_label_states_test[label])
                dic2_label_day_statistics_partdata[label] = statistic_separately(dic_day_test_states_partdata, dic_day_estimated_states_partdata, dic_day_accuracy_partdata)
                dic2_label_day_statistics_alldata[label] = statistic_separately(dic_day_test_states_alldata, dic_day_estimated_states_alldata, dic_day_accuracy_alldata)
            del label, filename_partdata, dic_day_test_states_partdata, dic_day_estimated_states_partdata, dic_day_accuracy_partdata, \
                filename_alldata, dic_day_test_states_alldata, dic_day_estimated_states_alldata, dic_day_accuracy_alldata
            output_separately('kNN_with_Lmnn_Partdata(k=%d, n=%d)' %(n_neighbors, n_train_lmnn), dic2_label_day_statistics_partdata, time.time()-t)
            output_separately('kNN_with_Lmnn_Alldata(k=%d, n=%d)' %(n_neighbors, n_train_lmnn), dic2_label_day_statistics_alldata, time.time()-t)
        del n_neighbors, dic2_label_day_statistics_partdata, dic2_label_day_statistics_alldata, t
    del n_train_lmnn

else: print('Strategy input error (together or separately).')
print('Finished. [' + time.strftime('%H:%M:%S') + ']')