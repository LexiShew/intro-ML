'''
hw1.py
Author: Lexi Shewchuk (ashewc01)

Tufts CS 135 Intro ML

'''

import numpy as np

def split_into_train_and_test(x_all_LF, frac_test=0.5, random_state=None):
    ''' Divide provided array into train and test set along first dimension

    User can provide a random number generator object to ensure reproducibility.

    Args
    ----
    x_all_LF : 2D array, shape = (n_total_examples, n_features) (L, F)
        Each row is a feature vector
    frac_test : float, fraction between 0 and 1
        Indicates fraction of all L examples to allocate to the "test" set
    random_state : np.random.RandomState instance or integer or None
        If int, code will create RandomState instance with provided value as seed
        If None, defaults to the current numpy random number generator np.random

    Returns
    -------
    x_train_MF : 2D array, shape = (n_train_examples, n_features) (M, F)
        Each row is a feature vector
        Should be a separately allocated array, NOT a view of any input array

    x_test_NF : 2D array, shape = (n_test_examples, n_features) (N, F)
        Each row is a feature vector
        Should be a separately allocated array, NOT a view of any input array

    Post Condition
    --------------
    This function should be side-effect free. The provided input array x_all_LF
    should not change at all (not be shuffled, etc.)

    Examples
    --------
    >>> x_LF = np.eye(10)
    >>> xcopy_LF = x_LF.copy() # preserve what input was before the call
    >>> train_MF, test_NF = split_into_train_and_test(
    ...     x_LF, frac_test=0.3, random_state=np.random.RandomState(0))
    >>> train_MF.shape
    (7, 10)
    >>> test_NF.shape
    (3, 10)
    >>> print(train_MF)
    [[0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
     [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
     [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]]
    >>> print(test_NF)
    [[0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
     [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]]

    ## Verify that input array did not change due to function call
    >>> np.allclose(x_LF, xcopy_LF)
    True

    References
    ----------
    For more about RandomState, see:
    https://stackoverflow.com/questions/28064634/random-state-pseudo-random-numberin-scikit-learn
    '''
    
    if random_state is None:
        random_state = np.random
        
    if type(random_state) is int:
        random_state = np.random.RandomState(random_state)
        
    num_data = x_all_LF.shape[0]          # array height -- number of subarrays
    num_test = int(np.ceil(frac_test * num_data))
    
    permuted_array = random_state.permutation(x_all_LF)
    x_test_NF = permuted_array[0:num_test:1]             # slice array
    x_train_MF = permuted_array[num_test:num_data:1] 
    
    return x_train_MF, x_test_NF


def calc_k_nearest_neighbors(data_NF, query_QF, K=1):
    ''' Compute and return k-nearest neighbors under Euclidean distance

    Any ties in distance may be broken arbitrarily.

    Args
    ----
    data_NF : 2D array, shape = (n_examples, n_features) aka (N, F)
        Each row is a feature vector for one example in dataset
    query_QF : 2D array, shape = (n_queries, n_features) aka (Q, F)
        Each row is a feature vector whose neighbors we want to find
    K : int, positive (must be >= 1)
        Number of neighbors to find per query vector

    Returns
    -------
    neighb_QKF : 3D array, (n_queries, n_neighbors, n_feats) (Q, K, F)
        Entry q,k is feature vector of the k-th neighbor of the q-th query
    '''
    
    Q = query_QF.shape[0]
    N = data_NF.shape[0]
    F = query_QF.shape[1]
    global neighb_QKF
    
    for q in range(0, Q):         # loop thru each vector in query      
        dist_dict = {}
        for n in range(0, N):     # compare the query vector to each data vector
            sum = 0
            for f in range(0, F): # compute distance
                sum += (query_QF[q][f]-data_NF[n][f]) * (query_QF[q][f]-data_NF[n][f])
            distance = np.sqrt(sum)
            dist_dict[n] = distance
            
        n_list = list(dict(sorted(dist_dict.items(), key=lambda x: x[1])).keys())
        while len(n_list) != K: # sort all the data vecs, then trim to only K
            n_list.pop(K)
            
        global K_array
        for i in range(len(n_list)): # put all neighbors in an array
            if i == 0:
                K_array = np.array([data_NF[n_list[i]]])
            else:
                K_array = np.append(K_array, data_NF[n_list[i]])
                
        if q == 0:                   # add each neighbor array to the return array
            neighb_QKF = np.array([K_array])
        else:
            neighb_QKF = np.append(neighb_QKF, K_array)
    
    return neighb_QKF.reshape(Q, K, F)
