import numpy as np
import scipy
from sklearn.svm import SVR
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.grid_search import RandomizedSearchCV
import math


def topic_from_LDA(X_train, X_test, n_topics, n_iter, alpha=0.1, eta=0.01):
    import lda
    X_train = X_train.astype(int)
    X_test = X_test.astype(int)

    lda_model = lda.LDA(n_topics, n_iter, alpha, eta, refresh=250)
    X_reduce_train = lda_model.fit_transform(X_train)
    X_reduce_test = lda_model.transform(X_test)
   
    return X_reduce_train, X_reduce_test, lda_model

def topic_from_LSI(X_train, X_test, n_topics):
    X_train_sparse = scipy.sparse.csr_matrix(X_train.T)
    U, S, V = scipy.sparse.linalg.svds(X_train_sparse)
    S_hat = np.diag(S[:n_topics])
    U_hat = U[:, :n_topics]
    S_hat_inverse = np.linalg.inv(S_hat)
    new_train = np.dot(S_hat_inverse, np.dot(U_hat.T, X_train.T))
    new_test = np.dot(S_hat_inverse, np.dot(U_hat.T, X_test.T))
    
    return new_train.T, new_test.T

def combine_extra_to_train(extra, train):
    if type(train) == scipy.sparse.csr.csr_matrix:
        return scipy.sparse.hstack([np.matrix(extra).T, train])
    else:
        return np.concatenate((np.matrix(extra).T, train), axis = 1)

def dtm_to_tf(X_train, X_test):
    transformer = TfidfTransformer(norm=None, use_idf = False)
    tf_train = transformer.fit_transform(X_train)
    tf_test = transformer.transform(X_test)
    return tf_train, tf_test

def dtm_to_tfidf(X_train, X_test):
    transformer = TfidfTransformer(norm=None, use_idf = True)
    tfidf_train = transformer.fit_transform(X_train)
    tfidf_test = transformer.transform(X_test)
    return tfidf_train, tfidf_test

def dtm_to_log1p(X_train, X_test):
    return np.log(X_train + 1), np.log(X_test + 1)

def involk_svr(X_total_train, Y_train, X_total_test, Y_test, C=math.pow(2,-10), tol=1e-5, epsilon=0.1, degree=1, gamma=1e-8):
    svr_poly = SVR(kernel='poly', C=C,epsilon=epsilon, degree=degree, gamma=gamma, tol=tol)
    svr_poly.fit(X_total_train, Y_train)
    result = svr_poly.predict(X_total_test)
    return metrics.mean_squared_error(result, Y_test)

def baseline(X_test_extra, Y_test):
    return metrics.mean_squared_error(X_test_extra, Y_test)

def optimize_svr(X_total_train, Y_train, X_total_test, Y_test, n_iter_search=30):
    '''
    svr = SVR()
    # params = [
    #     {'C': scipy.stats.expon(scale=1e-4), 'gamma': scipy.stats.expon(scale=1e-2), 'kernel' : ['rbf']},
    #     {'C': scipy.stats.expon(scale=1e-4), 'degree': [2, 3, 4, 5, 6], 'kernel' : ['poly']},
    #     {'C': scipy.stats.expon(scale=1e-4), 'kernel': ['linear']}
    # ]
    # params = {'C': scipy.stats.expon(scale=1e-4), 'degree': [1,2,3], 'kernel' : ['poly']}
    params = {'C': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-8, 1e-10], 
              'degree': [1, 2, 3], 
              'kernel' : ['poly'],
              'tol' : [1e-2, 1e-3, 1e-4, 1e-5],
              'epsilon' : [0.05, 0.1, 0.2, 0.5]}
    
    random_search = RandomizedSearchCV(svr, param_distributions=params, n_iter=n_iter_search)
    random_search.fit(X_total_train, Y_train)
    result = random_search.predict(X_total_test)
    
    mse = metrics.mean_squared_error(result, Y_test)
    # hyperparams = random_search.best_params_
    # return mse, random_search 
    '''    
    best_mse = -1
    best_param = {}
    for C in [1e-4, 1e-7, 1e-10]:
        for tol in [1e-5, 1e-3]:    
            for epsilon in [0.1, 0.5]:
                for degree in [1, 2, 3]:
                    for gamma in [1e-8, 1e-4, 0.1]:
                        mse = involk_svr(X_total_train, Y_train, X_total_test, Y_test, C, tol, epsilon, degree, gamma)
                        if mse < best_mse or best_mse == -1:
                            best_mse = mse
                            best_param['C'] = C
                            best_param['tol'] = tol
                            best_param['epsilon'] = epsilon
                            best_param['degree'] = degree
                            best_param['gamma'] = gamma
    return best_mse, best_param

