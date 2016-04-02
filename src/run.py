
from method import *
from read_data import *
import os
from tempfile import TemporaryFile
import pickle

os.chdir('/Users/hengweiguo/Documents/repo/volatility-prediction/src')


# X_train and X_test are doc-term matrices
# X_train_extra and X_test_extra are v-12 volatilities
# Y's are labels
X_train_extra, X_train, Y_train, X_test_extra, X_test, Y_test, vocab = generate_train_test_set(2006, 2, 0.2)

# # store the data
np.savez('train_test_data', X_train_extra=X_train_extra, X_train=X_train, Y_train=Y_train, X_test_extra=X_test_extra, X_test=X_test, Y_test=Y_test)
with open('vocab.pickle', 'w') as f:
    pickle.dump([vocab], f)

# # Getting back the objects:
npzfile = np.load('train_test_data.npz')
X_train_extra = npzfile['X_train_extra']
X_train = npzfile['X_train'] # doc term mat
Y_train = npzfile['Y_train'] # doc term mat
X_test_extra = npzfile['X_test_extra']
X_test = npzfile['X_test']
Y_test = npzfile['Y_test']
with open('vocab.pickle') as f:
    vocab= pickle.load(f)

n_topics = 20
n_iter = 200
X_train_lda, X_test_lda = topic_from_lda(X_train, X_test, n_topics, n_iter)

tf_train, tf_test = dtm_to_tf(X_train, X_test)
tfidf_train, tfidf_test = dtm_to_tfidf(X_train, X_test)
log1p_train, log1p_test = dtm_to_log1p(X_train, X_test)


# combine the doc-term data and the voliatility data
X_total_train_tf = combine_extra_to_train(X_train_extra, tf_train)
X_total_test_tf = combine_extra_to_train(X_test_extra, tf_test)

X_total_train_tfidf = combine_extra_to_train(X_train_extra, tfidf_train)
X_total_test_tfidf = combine_extra_to_train(X_test_extra, tfidf_test)

X_total_train_log1p = combine_extra_to_train(X_train_extra, log1p_train)
X_total_test_log1p = combine_extra_to_train(X_test_extra, log1p_test)

X_total_train_lda= combine_extra_to_train(X_train_extra, X_train_lda)
X_total_test_lda = combine_extra_to_train(X_test_extra, X_test_lda)


#-------------------------- Training and Testing ----------------------------

# train and test with the baseline: V-12
mse_V_minus_12 = baseline(X_test_extra, Y_test)

# train and test with TF+
mse_V_tf_plus = involk_svr(X_total_train_tf, Y_train, X_total_test_tf, Y_test)

# train and test with TFIDF+
mse_V_tfidf_plus = involk_svr(X_total_train_tfidf, Y_train, X_total_test_tfidf, Y_test)

# train and test with LDA
mse_V_lda = involk_svr(X_train_lda, Y_train, X_test_lda, Y_test)

# train and test with LDA+ volatility
mse_V_lda_plus = involk_svr(X_total_train_lda, Y_train, X_total_test_lda, Y_test)

# train and test with log1p+
mse_V_log1p_plus = involk_svr(X_total_train_log1p, Y_train, X_total_test_log1p, Y_test)

print(mse_V_minus_12)
print(mse_V_tf_plus)
print(mse_V_tfidf_plus)
print(mse_V_lda)
print(mse_V_lda_plus)
print(mse_V_log1p_plus)