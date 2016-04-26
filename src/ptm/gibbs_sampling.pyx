#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
import numpy as np
cimport numpy as np
import time
from libc.stdlib cimport malloc, free
from cython.operator cimport preincrement as inc, predecrement as dec

def gibbs_sampling(docs, int max_iter, word_topic, word_class,
                   int n_class, int[:,:] CW, int[:] sum_C, int[:,:] T, int[:] sum_T, int[:,:] TW, int[:,:] DT):

    for iter in range(max_iter):
        for di, doc in enumerate(docs):
            doc_topic = word_topic[di]
            doc_class = word_class[di]

            for si, sentence in enumerate(doc):
                len_sentence = len(sentence)

                sentence_topic = doc_topic[si]
                sentence_class = doc_class[si]

                for wi, word in enumerate(sentence):

                    if wi == 0:
                        prev_c = n_class
                    else:
                        prev_c = sentence_class[wi - 1]

                    if wi == len_sentence - 1:
                        next_c = n_class + 1
                    else:
                        next_c = sentence_class[wi + 1]

                    old_c = sentence_class[wi]
                    old_t = sentence_topic[wi]

                    # remove previous state
                    dec(CW[old_c, word])
                    dec(sum_C[old_c])
                    dec(T[prev_c, old_c])
                    dec(T[old_c, next_c])

                    # sample class
                    prob = (T[prev_c, :n_class] / np.asarray(T[prev_c]).sum()) \
                            * (T[:n_class, next_c] / np.sum(T[:n_class], 1))
                    prob[0] *= (TW[old_t, word] / sum_T[old_t])
                    prob[1:] *= np.asarray(CW[1:, word]) / np.asarray(sum_C[1:])

                    new_c = np.random.multinomial(1, prob).argmax()

                    sentence_class[wi] = new_c
                    inc(CW[new_c, word])
                    inc(sum_C[new_c])
                    inc(T[prev_c, new_c])
                    inc(T[new_c, next_c])

                    # remove previous topic state
                    dec(DT[di, old_t])
                    if old_c == 0:
                        dec(TW[old_t, word])
                        dec(sum_T[old_t])

                    # sample topic
                    prob = DT[di].copy()
                    if new_c == 0:
                        prob *= np.asarray(TW[:, word]) / np.asarray(sum_T)
                    prob /= np.sum(prob)

                    new_topic = np.random.multinomial(1, prob).argmax()
                    inc(DT[di, new_topic])
                    if new_c == 0:
                        inc(TW[new_topic, word])
                        inc(sum_T[new_topic])
                    sentence_topic[wi] = new_topic


# def _sample_topics(int[:] WS, int[:] DS, int[:] ZS, int[:, :] nzw, int[:, :] ndz, int[:] nz,
#                    double[:] alpha, double[:] eta, double[:] rands):
#     cdef int i, k, w, d, z, z_new
#     cdef double r, dist_cum
#     cdef int N = WS.shape[0]
#     cdef int n_rand = rands.shape[0]
#     cdef int n_topics = nz.shape[0]
#     cdef double eta_sum = 0
#     cdef double* dist_sum = <double*> malloc(n_topics * sizeof(double))
#     if dist_sum is NULL:
#         raise MemoryError("Could not allocate memory during sampling.")
#     with nogil:
#         for i in range(eta.shape[0]):
#             eta_sum += eta[i]
#
#         for i in range(N):
#             w = WS[i]
#             d = DS[i]
#             z = ZS[i]
#
#             dec(nzw[z, w])
#             dec(ndz[d, z])
#             dec(nz[z])
#
#             dist_cum = 0
#             for k in range(n_topics):
#                 # eta is a double so cdivision yields a double
#                 dist_cum += (nzw[k, w] + eta[w]) / (nz[k] + eta_sum) * (ndz[d, k] + alpha[k])
#                 dist_sum[k] = dist_cum
#
#             r = rands[i % n_rand] * dist_cum  # dist_cum == dist_sum[-1]
#             z_new = searchsorted(dist_sum, n_topics, r)
#
#             ZS[i] = z_new
#             inc(nzw[z_new, w])
#             inc(ndz[d, z_new])
#             inc(nz[z_new])
#
#         free(dist_sum)