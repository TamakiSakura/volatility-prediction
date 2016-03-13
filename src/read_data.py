import numpy as np
import textmining
import os

def read_logvol(logvol_filename):
    '''
    Take a filename, read the log volality in the file, and 
    return the numpy list of logvol ordered by the CUSIP number
    '''
    file_length = 0
    with open(logvol_filename) as logvol_file:
        current_line = logvol_file.readline()
        while current_line:
            file_length += 1
            current_line = logvol_file.readline()
    logvol = np.zeros(file_length)
    index = 0
    with open(logvol_filename) as logvol_file:
        current_line = logvol_file.readline()
        while current_line:
            curr_logvol = float(current_line.split()[0])
            logvol[index] = curr_logvol
            index += 1
            current_line = logvol_file.readline()
    return logvol

def construct_doc_term_matrix(tok_folders):
    '''
    Take a list of path to folders where tok.mda file is stored, 
    and return the numpy NumDoc x NumVocab matrix of doctermatrix
    Document ordered by CUSIP number
    Folders should not end with "/"
    Folders should be ordered by time
    '''
    tm_tdm = textmining.TermDocumentMatrix() 
    document_count = 0
    for tok_folder in tok_folders:
        tokfile_list = os.listdir(tok_folder)
        tokfile_list.sort()
        for tokfile_name in tokfile_list:
            with open(tok_folder + "/" + tokfile_name) as tokfile:
                tm_tdm.add_doc(tokfile.readline())
                document_count += 1
    np_tdm = 0 
    row_index = -1
    for row in tm_tdm.rows(cutoff=1):
        if row_index < 0:
            np_tdm = np.zeros(shape=(document_count, len(row)))
        else:
            np_tdm[row_index] = row
        row_index += 1
    return np_tdm
