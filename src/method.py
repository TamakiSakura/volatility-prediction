import numpy as np
from sklearn.svm import SVR
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfTransformer

def topic_from_lda(X_train, X_test, n_topics, n_iter):
    import lda
    
    lda_model = lda.LDA(n_topics, n_iter)
    X_reduce_train = lda_model.fit_transform(X_train)
    X_reduce_test = lda_model.transform(X_test)
    
    return X_reduce_train, X_reduce_test

def topic_from_LSI(X_train, X_test, n_topics):
    U, S, V = np.linalg.svd(X_train.T)
    S_hat = S[:n_topics, :n_topics]
    U_hat = U[:, :n_topics]
    S_hat_inverse = np.linalg.inv(S_hat)
    new_train = np.dot(S_hat_inverse, np.dot(U_hat.T, X_train.T)
    new_test = np.dot(S_hat_inverse, np.dot(U_hat.T, X_test.T)
    
    return new_train.T, new_test.T

def combine_extra_to_train(extra, train):
    return np.concatenate((np.matrix(extra).T, train), axis = 1)

def dtm_to_tf(X_train, X_test):
    transformer = TfidfTransformer(norm=None, use_idf = False)
    tfidf_train = transformer.fit_transform(X_train)
    tfidf_test = transformer.transform(X_test)
    return tfidf_train, tfidf_test

def dtm_to_tfidf(X_train, X_test):
    transformer = TfidfTransformer(norm=None, use_idf = True)
    tfidf_train = transformer.fit_transform(X_train)
    tfidf_test = transformer.transform(X_test)
    return tfidf_train, tfidf_test

def dtm_to_log1p(X_train, X_test):
    return np.log(X_train + 1), np.log(X_test + 1)

def involk_svr(X_total_train, Y_train, X_total_test, Y_test):
    svr_poly = SVR(kernel='poly', C=1e-5, degree=2)
    svr_poly.fit(X_total_train, Y_train)
    result = svr_poly.predict(X_total_test)
    return metrics.mean_squared_error(result, Y_test)

def baseline(Y_train, Y_test):
    return metrics.mean_squared_error(Y_train, Y_test)
