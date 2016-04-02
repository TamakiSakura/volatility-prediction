
from method import *
from read_data import *
import os
from tempfile import TemporaryFile
import pickle

os.chdir('/Users/hengweiguo/Documents/repo/volatility-prediction/src')


# # X_train and X_test are doc-term matrices
# # X_train_extra and X_test_extra are v-12 volatilities
# # Y's are labels
# X_train_extra, X_train, Y_train, X_test_extra, X_test, Y_test = generate_train_test_set(2006, 2, 0.2)
#
# # # store the data
# train_test_data = TemporaryFile()
# np.savez('train_test_data', X_train_extra=X_train_extra, X_train=X_train, Y_train=Y_train, X_test_extra=X_test_extra, X_test=X_test, Y_test=Y_test)
#

# # Getting back the objects:
npzfile = np.load('train_test_data.npz')
X_train_extra = npzfile['X_train_extra']
X_train = npzfile['X_train']
Y_train = npzfile['Y_train']
X_test_extra = npzfile['X_test_extra']
X_test = npzfile['X_test']
Y_test = npzfile['Y_test']


# combine the doc-term data and the voliatility data
X_total_train = combine_extra_to_train(X_train_extra, X_train)
X_total_test = combine_extra_to_train(X_test_extra, X_test)

# train and test with the baseline: SVR
mse = involk_svr(X_total_train, Y_train, X_total_test, Y_test)