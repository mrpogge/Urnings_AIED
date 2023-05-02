import numpy as np
import scipy.stats as sp
import scipy.special as sps
import itertools
import matplotlib.pyplot as plt


def binomial_gof(array, urn_size, true_p):
    a_len = len(array)
    elements_count = {}
    # iterating over the elements for frequency
    for element in array:
        # checking whether it is in the dict or not
        if element in elements_count:
            # incerementing the count by 1
            elements_count[element] += 1
        else:
            # setting the count to 1
            elements_count[element] = 1
    
    for i in range(urn_size):
        if i not in elements_count.keys():
            elements_count[i] = 0
    
    #the number of unique elements
    n = len(elements_count)
    khi_sq = 0

    for ky in range(len(elements_count)):
        x = list(elements_count.keys())[ky]
        print(x)
        expected_count = (sps.binom(urn_size + 1,x) * (true_p ** x) * ((1-true_p) ** (urn_size-x))) * a_len
        print(expected_count, elements_count[ky])

        #calculating test statistics
        khi_sq = khi_sq + ((elements_count[ky] - expected_count) ** 2 / expected_count)

    p = 1 - sp.chi2.cdf(khi_sq, n-1)
    return khi_sq, p

def frequency_check(array, set):
    elements_count = {}
    # iterating over the elements for frequency
    for element in array:
        # checking whether it is in the dict or not
        if element in elements_count:
            # incerementing the count by 1
            elements_count[element] += 1
        else:
            # setting the count to 1
            elements_count[element] = 1
    
    return elements_count


def all_binary_combination(window):

    lst = [list(i) for i in itertools.product([-1, 1], repeat=window)]
    binary_combinations = np.array(lst)
    return binary_combinations


#plt.hist(np.mean(all_binary_combination(10) * np.array([0,1,1,0,0,-1,-1,1,-1,0]), axis=1))

def MSE(col_means, true_value, change = None):
    mse =  np.sum((true_value - col_means) ** 2) / len(col_means)
    if change is not None:
        true_value = []
        for i in range(len(col_means)):
           true_value.append(0.5 + i * change)
        
        true_value = np.array(true_value)
        mse = np.sum((true_value - col_means) ** 2) / len(col_means)

    return mse

def coverage(urnings_matrix, true_value, change = None):

    n_sim = urnings_matrix.shape[1]
    col_lower = np.zeros(n_sim)
    col_upper = np.zeros(n_sim)
    coverage = 0
    for i in range(n_sim):
        confint = np.percentile(urnings_matrix[:,i] , [5, 95])
        col_lower[i] = confint[0]
        col_upper[i] = confint[1]
        #calculating coverage
        if confint[0] < true_value < confint[1]:
            coverage += 1

    coverage = coverage / n_sim

    if change is not None:
        coverage = 0
        true_value = 0.5
        for i in range(n_sim):
            if confint[0] < true_value < confint[1]:
                coverage += 1
            true_value += change
    
    coverage = coverage / n_sim
    
    return coverage

def hitting_time(col_means, true_value, tol = 0.01):
    SE_true_value = tol
    #print out the iteration when the mean was in this error radius
    error_radius_logical = []
    
    for clm in col_means:
        error_radius_logical.append((true_value - SE_true_value) < clm < (true_value + SE_true_value))
    
    trues_vec = np.where(error_radius_logical)[0]
    if len(trues_vec) > 0:
        hitting_time = np.min(trues_vec)
    else:
        hitting_time = 999
    
    return hitting_time

def hitting_below(col_means, true_value):
    dropped_below = []
    for clm in col_means:
        dropped_below.append(clm < true_value)
    
    trues_vec = np.where(dropped_below)[0]
    if len(trues_vec) > 0:
        hitting_time = np.min(trues_vec)
    else:
        hitting_time = 999
    
    return hitting_time

def pRasch(u,v,n,m):
    return ((u*(m-v))/(u*(m-v)+(n-u)*v))

