import numpy as np

def read_logvol(logvol_filename):
    '''
    Take a filename, read the log volality in the file, 
    and return the list ordered by the CUSIP number
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

def construct_doc_term_matrix():
