
from method import *
from read_data import *
import os
import pickle
import time
from sklearn.preprocessing import scale
from ptm import HMM_LDA

# os.chdir('/Users/hengweiguo/Documents/repo/volatility-prediction/src')
#
#
# # X_train and X_test are doc-term matrices
# # X_train_extra and X_test_extra are v-12 volatilities
# # Y's are labels
# X_train_extra, X_train, Y_train, X_test_extra, X_test, Y_test, vocab, indices = generate_train_test_set(2006, 2, 0.01)
# # # store the data
# np.savez('train_test_data_001_20topics', X_train_extra=X_train_extra, X_train=X_train, Y_train=Y_train, X_test_extra=X_test_extra, X_test=X_test, Y_test=Y_test, indices=indices)
# # save vocab seperately since it's not np array
# with open('vocab_001.pickle', 'w') as f:
#     pickle.dump([vocab], f)
#
# # do the lda reduction
# n_topics = 20
# n_iter = 1000
# t0 = time.time()
# X_train_lda, X_test_lda, lda_model = topic_from_lda(X_train, X_test, n_topics, n_iter)
# t1 = time.time()
# print('lda takes time: ' + str(t1 - t0))
# np.savez('lda_data_001_20topics', X_train_lda=X_train_lda, X_test_lda=X_test_lda)
#
# with open('lda_model_001_20topics.pickle', 'w') as f:
#     pickle.dump([lda_model], f)


# Getting back the objects:
npzfile = np.load('train_test_data_001_20topics.npz')
X_train_extra = npzfile['X_train_extra']
X_train = npzfile['X_train'] # doc term mat
Y_train = npzfile['Y_train'] # doc term mat
X_test_extra = npzfile['X_test_extra']
X_test = npzfile['X_test']
Y_test = npzfile['Y_test']
indices = npzfile['indices']

npzfile = np.load('lda_data_001_20topics.npz')
X_train_lda = npzfile['X_train_lda']
X_test_lda = npzfile['X_test_lda']

#with open('vocab_001.pickle') as f:
#    vocab= pickle.load(f)
    
with open('lda_model_001_20topics.pickle') as f:
    lda_model= pickle.load(f)[0]


# # print the lda topics
# n_top_words = 12
# topic_word = lda_model.topic_word_
# for i, topic_dist in enumerate(topic_word):
#     topic_words = np.array(vocab).T[np.argsort(topic_dist)][:-(n_top_words+1):-1]
#     topic_words = topic_words.tolist()
#     topic_words = [item for sublist in topic_words for item in sublist if item not in ['and', 'in', 'the', 'of', 'a', 'to', 'is', 'we', 'that', 'for']]
#     print('Topic {}: {}'.format(i, ' '.join(topic_words)))

# hmm_LDA_vocab, hmm_lda_corpus, test_count = generateDataForHmmLDA(2006, 2, indices)
# hmm_lda_X_train = hmm_lda_corpus[:-test_count]
# hmm_LDA_X_test = hmm_lda_corpus[-test_count:]
#
# # train hmm lda
# n_docs = len(hmm_lda_X_train)
# n_voca = len(hmm_LDA_vocab)
# n_topic = 20
# n_class = 20
# max_iter = 20
# alpha = 0.1
# beta = 0.01
# gamma = 0.1
# eta = 0.1
# model = HMM_LDA(n_docs, n_voca, n_topic, n_class, alpha=alpha, beta=beta, gamma=gamma, eta=eta, verbose=True)
# model.fit(hmm_lda_X_train, max_iter=max_iter)
#
# with open('hmm_lda_001_20topics_20iters.pickle', 'w') as f:
#     pickle.dump([hmm_LDA_vocab, hmm_lda_corpus, hmm_lda_X_train, hmm_LDA_X_test, model], f)

with open('hmm_lda_001_20topics_20iters.pickle') as f:
    tmpData = pickle.load(f)
    vocab_hmmlda = tmpData[0]
    corpus_hmmlda = tmpData[1]  # hmm_lda_X_train + hmm_LDA_X_test
    X_train_hmmlda = tmpData[2]
    X_test_hmmlda = tmpData[3]
    model = tmpData[4]
    del tmpData

tf_train, tf_test = dtm_to_tf(X_train, X_test)
tfidf_train, tfidf_test = dtm_to_tfidf(X_train, X_test)
log1p_train, log1p_test = dtm_to_log1p(X_train, X_test)

scaleData = 1
if scaleData:
    # can't do with_mean=True with sparse matrix
    tf_train = scale(tf_train, with_mean=False)
    tf_test = scale(tf_test, with_mean=False)
    tfidf_train = scale(tfidf_train, with_mean=False)
    tfidf_test = scale(tfidf_test, with_mean=False)
    log1p_train = scale(log1p_train, with_mean=False)
    log1p_test = scale(log1p_test, with_mean=False)
    X_train_lda = scale(X_train_lda, with_mean=False)
    X_test_lda = scale(X_test_lda, with_mean=False)

# combine the doc-term data and the voliatility data
# X_total_train_tf = combine_extra_to_train(X_train_extra, tf_train)
# X_total_test_tf = combine_extra_to_train(X_test_extra, tf_test)

# X_total_train_tfidf = combine_extra_to_train(X_train_extra, tfidf_train)
# X_total_test_tfidf = combine_extra_to_train(X_test_extra, tfidf_test)

# X_total_train_log1p = combine_extra_to_train(X_train_extra, log1p_train)
# X_total_test_log1p = combine_extra_to_train(X_test_extra, log1p_test)

# X_total_train_lda= combine_extra_to_train(X_train_extra, X_train_lda)
# X_total_test_lda = combine_extra_to_train(X_test_extra, X_test_lda)


# Code for predicting difference
Y_train = Y_train - X_train_extra
Y_test = Y_test - X_test_extra
X_test_extra = np.zeros(len(X_test_extra))

# -------- Find thehyper parameters ----------
# n_iter_search = 20
# t0 = time.time()
# mse_tf_plus, random_search_tf_plus = optimize_svr(X_total_train_tf, Y_train, X_total_test_tf, Y_test, n_iter_search)
# t1 = time.time()
# print('tune hyper parameter for tf+ takes time: ' + str(t1 - t0))
#-------------------------- Training and Testing ----------------------------


# train and test with the baseline: V-12
t0 = time.time()
mse_V_minus_12 = baseline(X_test_extra, Y_test)
t1 = time.time()
print('mse_V_minus_12 takes time: ' + str(t1 - t0))
print('mse_V_minus_12: ' + str(mse_V_minus_12))

# train and test with TF+
t0 = time.time()
#mse_V_tf_plus = involk_svr(X_total_train_tf, Y_train, X_total_test_tf, Y_test)
mse_V_tf_plus, params = optimize_svr(X_total_train_tf, Y_train, X_total_test_tf, Y_test)
t1 = time.time()
print('mse_V_tf_plus takes time: ' + str(t1 - t0))
print('mse_V_tf_plus: ' + str(mse_V_tf_plus))
print('SVR params: '+ str(params))

# train and test with TFIDF+
t0 = time.time()
#mse_V_tfidf_plus = involk_svr(X_total_train_tfidf, Y_train, X_total_test_tfidf, Y_test)
mse_V_tfidf_plus, params = optimize_svr(X_total_train_tfidf, Y_train, X_total_test_tfidf, Y_test)
t1 = time.time()
print('mse_V_tfidf_plus takes time: ' + str(t1 - t0))
print('mse_V_tfidf_plus: ' + str(mse_V_tfidf_plus))
print('SVR params: '+ str(params))

# train and test with LDA
t0 = time.time()
#mse_V_lda = involk_svr(X_train_lda, Y_train, X_test_lda, Y_test)
mse_V_lda, params = optimize_svr(X_train_lda, Y_train, X_test_lda, Y_test)
t1 = time.time()
print('mse_V_lda takes time: ' + str(t1 - t0))
print('mse_V_lda: ' + str(mse_V_lda))
print('SVR params: '+ str(params))

# train and test with HMM-LDA
# t0 = time.time()
#mse_V_lda = involk_svr(X_train_hmmlda, Y_train, X_test_hmmlda, Y_test)
# mse_V_lda, params = optimize_svr(X_train_hmmlda, Y_train, X_test_hmmlda, Y_test)
# t1 = time.time()
# print('mse_V_hmmlda takes time: ' + str(t1 - t0))
# print('mse_V_hmmlda: ' + str(mse_V_lda))
# print('SVR params: '+ str(params))

# train and test with LDA+ volatility
# 0 = time.time()
# mse_V_lda_plus = involk_svr(X_total_train_lda, Y_train, X_total_test_lda, Y_test)
# t1 = time.time()
# print('mse_V_lda_plus takes time: ' + str(t1 - t0))
# print('mse_V_lda_plus: ' + str(mse_V_lda_plus))

# train and test with log1p+
# t0 = time.time()
#mse_V_log1p_plus = involk_svr(X_total_train_log1p, Y_train, X_total_test_log1p, Y_test)
#t1 = time.time()
#print('mse_V_log1p_plus takes time: ' + str(t1 - t0))
#print('mse_V_log1p_plus: ' + str(mse_V_log1p_plus))
