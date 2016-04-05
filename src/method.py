import numpy as np
import scipy
from sklearn.svm import SVR
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.grid_search import RandomizedSearchCV

def topic_from_lda(X_train, X_test, n_topics, n_iter):
    import lda
    X_train = X_train.astype(int)
    X_test = X_test.astype(int)

    lda_model = lda.LDA(n_topics, n_iter)
    X_reduce_train = lda_model.fit_transform(X_train)
    X_reduce_test = lda_model.transform(X_test)
    
    return X_reduce_train, X_reduce_test, lda_model

def topic_from_LSI(X_train, X_test, n_topics):
    U, S, V = np.linalg.svd(X_train.T)
    S_hat = S[:n_topics, :n_topics]
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

def involk_svr(X_total_train, Y_train, X_total_test, Y_test, C=1e-5, degree=2):
    svr_poly = SVR(kernel='poly', C=C, degree=degree)
    svr_poly.fit(X_total_train, Y_train)
    result = svr_poly.predict(X_total_test)
    return metrics.mean_squared_error(result, Y_test)

def baseline(X_test_extra, Y_test):
    return metrics.mean_squared_error(X_test_extra, Y_test)

def optimize_svr(X_total_train, Y_train, X_total_test, Y_test, n_iter_search):
    svr = SVR()
    # params = [
    #     {'C': scipy.stats.expon(scale=1e-4), 'gamma': scipy.stats.expon(scale=1e-2), 'kernel' : ['rbf']},
    #     {'C': scipy.stats.expon(scale=1e-4), 'degree': [2, 3, 4, 5, 6], 'kernel' : ['poly']},
    #     {'C': scipy.stats.expon(scale=1e-4), 'kernel': ['linear']}
    # ]
    params = {'C': scipy.stats.expon(scale=1e-4), 'degree': [1,2,3], 'kernel' : ['poly']}

    
    random_search = RandomizedSearchCV(svr, param_distributions=params, n_iter=n_iter_search)
    random_search.fit(X_total_train, Y_train)
    result = random_search.predict(X_total_test)
    
    mse = metrics.mean_squared_error(result, Y_test)
   # hyperparams = random_search.best_params_
    # return mse, random_search 
    return mse, random_search
