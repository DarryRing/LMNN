import sklearn
from sklearn import datasets
from sklearn import neighbors
from sklearn import decomposition
#from pprint import pprint
import metric_learn #Download: https://pypi.python.org/pypi/metric-learn/
import scipy
import time
import numpy
import copy

candidate_neighbors = [3, 5]
n_components = 100
strategy = 'step-by-step' #'sorted' or 'step-by-step'
max_iter = 1000 #default:1000, for test

if strategy == 'sorted': from_top_n_categories = 2

dic_index_category = {}
dic_index_count = {}
train_news = sklearn.datasets.fetch_20newsgroups(subset='train')
target_names = train_news.target_names
for i in range(len(target_names)):
    dic_index_category[i] = target_names[i]
    dic_index_count[i] = 0
del i
for index in train_news.target:
    dic_index_count[index] += 1
del index
sorted_index_count = sorted(dic_index_count.items(), key=lambda x:x[1], reverse=True)#descending order

#output categories (ranking by count) to file
file = open('20newsCategory.csv', 'w')#, encoding='shift_jis'
file.write("Ranking, Category, Number of texts\n")
for i in range(len(sorted_index_count)):
    file.write('%d,' %i + dic_index_category[sorted_index_count[i][0]] + ',%d\n' %sorted_index_count[i][1])
del i
file.close()
file = open('20newsResult_step_allLabel.csv', 'w')
file.write('Method,Accuracy\n')
file.close()
file = open('20newsResult_step_eachLabel.csv', 'w')
file.close()

#change: test_data, test_labels; use: test_vectors, test_labels
def step_classification(n_neighbors, cur_label, train_vectors, train_binary_labels, test_data, test_vectors, test_labels):
    classifier = neighbors.KNeighborsClassifier(n_neighbors)
    classifier.fit(train_vectors, train_binary_labels)
    test_true_binary_labels = []
    for i in range(len(test_labels)):
        test_true_binary_labels.append(1 if test_labels[i] == cur_label else 0)
    del i
    test_estimated_binary_labels = classifier.predict(test_vectors)
    delete_index = 0
    n_right = 0
    n_estimated = 0
    #print('\tNumber of vectors: %d. Number of labels: %d.' %(len(test_estimated_binary_labels),len(test_true_binary_labels)))
    for i in range(len(test_estimated_binary_labels)):
        if test_estimated_binary_labels[i] == 1:
            n_estimated += 1
            if test_true_binary_labels[i] == 1: 
                n_right += 1
            del test_data[delete_index]
            test_labels = numpy.delete(test_labels, delete_index)
            delete_index -= 1
        delete_index += 1
    del i, delete_index
    return (n_right, test_data, test_labels, n_estimated) 
    
def svd_transform(train_vectors, test_vectors):
    svd = decomposition.TruncatedSVD(n_components=n_components, algorithm='arpack')
    all_vectors = scipy.sparse.vstack([train_vectors, test_vectors])#join vectors
    svd.fit(all_vectors)
    del all_vectors
    train_vectors_svd = svd.transform(train_vectors)
    test_vectors_svd = svd.transform(test_vectors)
    return (train_vectors_svd, test_vectors_svd)
    
def output_step(output):
    file = open('20newsResult_step_eachLabel.csv', 'a')
    file.write(output)
    file.close()
    print(output)

if strategy == 'step-by-step':
    #test_true_labels = datasets.fetch_20newsgroups(subset='test').target
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(vocabulary=sklearn.feature_extraction.text.TfidfVectorizer().fit(datasets.fetch_20newsgroups(subset='all').data).get_feature_names())
    test_news = datasets.fetch_20newsgroups(subset='test')
    train_vectors = vectorizer.fit_transform(train_news.data) 
    train_labels = train_news.target
    for n_neighbors in candidate_neighbors:
        output_step('k=%d\n' %n_neighbors)
        dic_test_data = {}
        dic_test_data['knn'] = copy.deepcopy(test_news.data)
        dic_test_data['knn_svd'] = copy.deepcopy(test_news.data)
        dic_test_data['knn_svd_lmnn'] = copy.deepcopy(test_news.data)
        dic_test_labels = {}
        dic_test_labels['knn'] = copy.deepcopy(test_news.target)
        dic_test_labels['knn_svd'] = copy.deepcopy(test_news.target)
        dic_test_labels['knn_svd_lmnn'] = copy.deepcopy(test_news.target)
        dic_n_right = {}
        dic_n_right['knn'] = 0
        dic_n_right['knn_svd'] = 0
        dic_n_right['knn_svd_lmnn'] = 0
        accuracy = {}
        for i_label in range(0, len(sorted_index_count)-1):
            cur_label = sorted_index_count[i_label][0]
            output_step('The %dth label: %d ' %(i_label + 1, cur_label) + dic_index_category[cur_label] + '\n')
            train_binary_labels = []
            for i in range(len(train_labels)):
                train_binary_labels.append(1 if train_labels[i] == cur_label else 0)
            del i
            
            #knn
            startTime = time.time()
            output_step('\tknn (rest: %d): \n' %len(dic_test_labels['knn']))
            test_vectors = vectorizer.fit_transform(dic_test_data['knn'])
            result = step_classification(n_neighbors, cur_label, train_vectors, train_binary_labels, dic_test_data['knn'], test_vectors, dic_test_labels['knn'])
            del test_vectors
            dic_n_right['knn'] += result[0]
            dic_test_data['knn'] = result[1]
            dic_test_labels['knn'] = result[2]
            output_step('\t\tresult: %d/%d, %ds\n' %(result[0], result[3], time.time() - startTime))
            del startTime
            
            #knn+svd
            startTime = time.time()
            output_step('\tknn_svd (rest: %d): \n' %len(dic_test_labels['knn_svd']))
            test_vectors = vectorizer.fit_transform(dic_test_data['knn_svd'])
            (train_vectors_svd, test_vectors_svd) = svd_transform(train_vectors, test_vectors)
            del test_vectors
            result = step_classification(n_neighbors, cur_label, train_vectors_svd, train_binary_labels, dic_test_data['knn_svd'], test_vectors_svd, dic_test_labels['knn_svd'])
            del train_vectors_svd, test_vectors_svd
            dic_n_right['knn_svd'] += result[0]
            dic_test_data['knn_svd'] = result[1]
            dic_test_labels['knn_svd'] = result[2]
            output_step('\t\tresult: %d/%d, %ds\n' %(result[0], result[3], time.time() - startTime))
            del startTime
            
            #knn_svd_lmnn
            startTime = time.time()
            output_step('\tknn_svd_lmnn (rest: %d): \n' %len(dic_test_labels['knn_svd_lmnn']))
            test_vectors = vectorizer.fit_transform(dic_test_data['knn_svd_lmnn'])
            (train_vectors_svd, test_vectors_svd) = svd_transform(train_vectors, test_vectors)
            del test_vectors
            lmnn = metric_learn.LMNN(k=n_neighbors, max_iter=max_iter)
            lmnn.fit(train_vectors_svd, numpy.asarray(train_binary_labels))
            train_vectors_svd_lmnn = lmnn.transform(train_vectors_svd)
            del train_vectors_svd
            test_vectors_svd_lmnn = lmnn.transform(test_vectors_svd)
            del test_vectors_svd
            result = step_classification(n_neighbors, cur_label, train_vectors_svd_lmnn, train_binary_labels, dic_test_data['knn_svd_lmnn'], test_vectors_svd_lmnn, dic_test_labels['knn_svd_lmnn'])
            del train_vectors_svd_lmnn, test_vectors_svd_lmnn
            dic_n_right['knn_svd_lmnn'] += result[0]
            dic_test_data['knn_svd_lmnn'] = result[1]
            dic_test_labels['knn_svd_lmnn'] = result[2]
            output_step('\t\tresult: %d/%d, %ds\n' %(result[0], result[3], time.time() - startTime))
            del startTime
            
            del train_binary_labels
        del i_label, cur_label
        accuracy['knn'] = float(dic_n_right['knn'])/float(len(test_news.target))
        accuracy['knn_svd'] = float(dic_n_right['knn_svd'])/float(len(test_news.target))
        accuracy['knn_svd_lmnn'] = float(dic_n_right['knn_svd_lmnn'])/float(len(test_news.target))
        #write to file
        file = open('20newsResult_step_allLabel.csv', 'a')
        file.write('knn (%d),%f\nknn_svd (%d),%f\nknn_svd_lmnn,%f\n' %(n_neighbors, accuracy['knn'], n_components, accuracy['knn_svd'], accuracy['knn_svd_lmnn']))
        file.close()
    del n_neighbors

if strategy == 'sorted':
    file = open('20newsResult.csv', 'w')#, encoding='shift_jis'
    file.write('No. of labels,Method,Accuracy,Time (s)\n')
    file.close()
    
    for categories_number in range(from_top_n_categories, len(sorted_index_count)):
        categories = []
        for i in range(categories_number):
            categories.append(dic_index_category[sorted_index_count[i][0]])
        del i
        #pprint(newsgroups_all.target_names)
        
        #obtain common vocabulary of train and test (all categories->173762)
        t = time.time()
        print("Transform texts to vectors.")
        vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(vocabulary=sklearn.feature_extraction.text.TfidfVectorizer().fit(datasets.fetch_20newsgroups(subset='all', categories=categories).data).get_feature_names())
        
        #obtain training data
        train_news = datasets.fetch_20newsgroups(subset='train', categories=categories)
        train_vectors = vectorizer.fit_transform(train_news.data)#document->vector(tf-idf)
        #print("Average number of no-zero elements in each train text: %f." % (train_vectors.nnz / float(train_vectors.shape[0])))#average number of no-zero elements
        train_labels = train_news.target
        
        #obtain testing data
        test_news = datasets.fetch_20newsgroups(subset='test', categories=categories)
        test_vectors = vectorizer.fit_transform(test_news.data)
        #print("Average number of no-zero elements in each test text: %f." % (test_vectors.nnz / float(test_vectors.shape[0])))#average number of no-zero elements
        test_labels = test_news.target
        print("\tdone in %fs." % (time.time() - t))
        
        for n_neighbors in candidate_neighbors:
            #without lmnn
            t = time.time()
            classifier = neighbors.KNeighborsClassifier(n_neighbors)
            classifier.fit(train_vectors, train_labels)
            accuracy = classifier.score(test_vectors, test_labels)
            print("Accuracy of kNN (k=%d) (%d categories): %f." %(n_neighbors, categories_number, accuracy))
            print("\tdone in %fs." % (time.time() - t) + '[' + time.strftime('%H:%M:%S') + ']')
            file = open('20newsResult.csv', 'a')#, encoding='shift_jis'
            file.write('%d,kNN(k=%d),%f,%d\n' %(categories_number, n_neighbors, accuracy, time.time() - t))
            file.close()
            
            #svd without lmnn
            t = time.time()
            svd = decomposition.TruncatedSVD(n_components=n_components, algorithm='arpack')
            all_vectors = scipy.sparse.vstack([train_vectors, test_vectors])#join vectors
            svd.fit(all_vectors)
            train_vectors_changed = svd.transform(train_vectors)
            test_vectors_changed =  svd.transform(test_vectors)
            classifier.fit(train_vectors_changed, train_labels)
            accuracy = classifier.score(test_vectors_changed, test_labels)
            print("Accuracy of kNN (k=%d) + svd (n=%d) (%d categories): %f." %(n_neighbors, n_components, categories_number, accuracy))
            print("\tdone in %fs." % (time.time() - t) + '[' + time.strftime('%H:%M:%S') + ']')
            file = open('20newsResult.csv', 'a')#, encoding='shift_jis'
            file.write('%d,kNN(k=%d)_svd(n=%d),%f,%d\n' %(categories_number, n_neighbors, n_components, accuracy, time.time() - t))
            file.close()
            
            #lmnn
            t = time.time()
            lmnn = metric_learn.LMNN(k=n_neighbors)#, max_iter=max_iter
            lmnn.fit(train_vectors_changed, train_labels)
            train_vectors_changed = lmnn.transform(train_vectors_changed)
            test_vectors_changed = lmnn.transform(test_vectors_changed)
            classifier.fit(train_vectors_changed, train_labels)
            accuracy = classifier.score(test_vectors_changed, test_labels)
            print("Accuracy of kNN (k=%d) + svd (n=%d) + lmnn (%d categories): %f." %(n_neighbors, n_components, categories_number, accuracy))
            print("\tdone in %fs." % (time.time() - t) + '[' + time.strftime('%H:%M:%S') + ']')
            file = open('20newsResult.csv', 'a')#, encoding='shift_jis'
            file.write('%d,kNN(k=%d)_svd(n=%d)_lmnn,%f,%d\n' %(categories_number, n_neighbors, n_components, accuracy, time.time() - t))
            file.close()
        del n_neighbors

print("Finished.")