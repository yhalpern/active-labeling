import numpy as np
import random
import matplotlib.pyplot as plt
import sys
#from cvxopt import matrix, solvers
import sklearn.metrics as metrics
from scipy.optimize import  fmin_l_bfgs_b
import h5py
import bisect

def argmax(L):
    return max(enumerate(L), key=lambda x: x[1])[0]

def argmin(L):
    return min(enumerate(L), key=lambda x: x[1])[0]

#settings
K = 25 #label K at a time
random.seed(123)
#solvers.options['show_progress'] = False
#########

def count_all_swaps(bins):
    print 'debug: counting all swaps'
    N = len(bins)
    swaps = 0
    for i in xrange(N):
        swaps += bins[i].swaps
        print 'internal swaps', i, bins[i].swaps
        for j in xrange(i+1,N):
            print i, 'swaps with', j,  bins[i].pos*bins[j].neg
            swaps += bins[i].pos*bins[j].neg
    print 'total', swaps
    return swaps


def do_labeling(id):
    return int(Y_pool[id])

def count_swaps(labels):
    total = len(labels)
    pos = sum(labels)
    neg = total - pos
    if pos == 0 or neg == 0:
        return 0
    auc = metrics.roc_auc_score(labels,xrange(total))
    #swaps = 0
    #for i in xrange(total):
    #    if labels[i] == 1:
    #        for j in xrange(i+1, total):
    #            if labels[j] == 0:
    #                swaps += 1
    #return swaps
    return (1-auc)*pos*neg

def emplace_bins(all_bins, new_bins):
    for b in new_bins:
        bisect.insort_left(all_bins, b)

def select_bin(bins, mode='bounds'):
    if mode == 'gap':
        gap_vals = [0]*len(bins)
        bounds,x0 = build_bounds(bins)
        for i,b in enumerate(bins):
            low,high = bounds[i]

            b.p = low
            h = high_bound(bins)
            l = low_bound(bins)
            gap_val_low = h-l
            
            b.p = high
            h = high_bound(bins)
            l = low_bound(bins)
            gap_val_high = h-l

            b.p = None
            gap_vals[i] = min(gap_val_low, gap_val_high)

        idx = argmin(gap_vals)
        choice = 1

    if mode == 'bounds':
        bounds,x0 = build_bounds(bins)
        idx = argmax([[h-l] for l,h in bounds])
        print 'choosing a bin with bounds', bounds[idx]
        if idx == 0:
            choice = 0
        elif idx == len(bins)-1:
            choice = 2
        else:
            choice = 1

    if mode == 'largest':
        #selects the largest bin
        sizes = [b.size if b.labels is None else -1 for b in bins]
        idx = argmax(sizes)
        if sizes[idx] == -1: #everything is labeled
            return None
        choice = 1
    
    my_bin = bins.pop(idx)

    return my_bin, choice
class Bin:
    def __init__(self, pop):
        # pop is a sorted list of integer ids
        self.boundaries = (pop[0], pop[-1])
        self.size = len(pop)
        self.p = None
        self.swaps = None
        self.labels = None
        self.pos = None
        self.neg = None
        self.auc= None
        self.members = pop

    def __lt__(self, other):
        return self.boundaries[0] < other.boundaries[0]

    def __str__(self):
        my_str = []
        my_str.append('Bin: '+str(self.boundaries[0])+'-'+str(self.boundaries[1]))
        my_str.append('P: '+str(self.p))
        my_str.append('swaps: '+ str(self.swaps))
        my_str.append('auc: '+ str(self.auc))
        my_str.append('pos: '+str( self.pos))
        my_str.append('neg: '+str(self.neg))
        return '\n'.join(my_str)

    def label(self):
        labels = map(do_labeling, self.members)
        self.pos = sum(labels)
        self.neg = self.size - self.pos
        self.p = float(self.pos) / self.size
        self.swaps = count_swaps(labels)
        if self.pos == 0 or self.neg == 0:
            self.auc = 1
        else:
            self.auc = 1-self.swaps / (self.pos*self.neg)
        self.labels = labels

    def split(self, where):
        if len(self.members) <= K:
            return self, []

        if where == 0: #return bottom
            low_bin = Bin(self.members[:K])
            high_bin = Bin(self.members[K:])
            return low_bin, [high_bin]
        elif where == 2: #return top
            low_bin = Bin(self.members[:-K])
            high_bin = Bin(self.members[-K:])
            return high_bin, [low_bin]
        elif where == 1: #return middle
            midpoint = self.size / 2
            low_bin = Bin(self.members[:midpoint - K/2])
            mid_bin = Bin(self.members[midpoint-K/2:midpoint+K/2])
            high_bin = Bin(self.members[midpoint+K/2:])
            return mid_bin, (low_bin, high_bin)

def build_bounds(bins):
    bounds = []
    x0 = []
    N = len(bins)
    for i,b in enumerate(bins):
        if b.p is not None:
            ub = b.p
            lb = b.p
        else:
            if i == len(bins)-1:
                ub = 1
            else:
                ub = bins[i+1].p
            if i == 0:
                lb = 0
            else:
                lb = bins[i-1].p
            assert ub is not None
            assert lb is not None
            if lb > ub:
                lb, ub = ub, lb #swap

        bounds.append((lb, ub))
        x0.append((lb + ub)/2)

    return bounds, x0

def high_bound(bins):
    N = len(bins) #total bins
    L = len([b for b in bins if b.labels is not None]) #labeled bins
    U = N - L #unlabeled bins

    def f(p):
        swaps = 0
        for i in xrange(N):
            if bins[i].swaps is None:
                swaps += 0.5*bins[i].size**2 *p[i]*(1-p[i])
            else:
                swaps += bins[i].swaps

            for j in xrange(i+1,N):
                swaps += bins[i].size*p[i]*bins[j].size*(1-p[j])
        return -swaps
    
    bounds,x0 = build_bounds(bins)
    res = fmin_l_bfgs_b(f, x0, fprime=None, approx_grad=1, bounds=bounds)
    return -res[1]
    #x = [min(b) for b in bounds]
    #return -f(x)




def low_bound(bins):
    N = len(bins) #total bins
    L = len([b for b in bins if b.labels is not None]) #labeled bins
    U = N - L #unlabeled bins

    def f(p):
        swaps = 0
        for i in xrange(N):
            if bins[i].swaps is None:
                swaps += 0
            else:
                swaps += bins[i].swaps

            for j in xrange(i+1,N):
                swaps += bins[i].size*p[i]*bins[j].size*(1-p[j])
        return swaps
    
    bounds,x0 = build_bounds(bins)
    res = fmin_l_bfgs_b(f, x0, fprime=None, approx_grad=1, bounds=bounds)
    return res[1]
    #x = [max(b) for b in bounds]
    #return f(x)

dataset_storage = h5py.File('data.h5py', 'r')
Yhat_pool = dataset_storage['Yhat'][...]
Y_pool = dataset_storage['Y'][...]
dataset_storage.close()
print 'Yhat', Yhat_pool[:5]
print 'Y', Y_pool[:5]

all_patients = Bin(range(len(Y_pool)))
total_pop = float(all_patients.size)

all_patients.label()
all_patients.labels = None
all_patients.p = None
true_swaps = all_patients.swaps 
pos = all_patients.pos
neg = all_patients.neg
print all_patients
bins = [all_patients]

X = []
high = []
low = []
true = []
l = 0
h = float('inf')
for i in xrange(32):
    current_bin,where = select_bin(bins)
    print 'selecting', current_bin

    if current_bin is None:
        break

    active_bin, other_bins = current_bin.split(where)
    active_bin.label()
    emplace_bins(bins, [active_bin])
    emplace_bins(bins, other_bins)
    h = min(h,high_bound(bins))
    l = max(l,low_bound(bins))
    high.append(1-h/float(pos*neg))
    low.append(1-l/float(pos*neg))
    true.append(1-true_swaps/float(pos*neg))
    print 'high, true, low', h, true_swaps, l, 'gap', h-l
    X.append(i)

plt.plot(X,high, 'k--', label='upper-bound')
plt.plot(X,low, 'k--', label='lower-bound')
plt.plot(X,true, 'r-', label='true AUC')

pop = range(len(Y_pool))
random.shuffle(pop)

estimates = []
sample = []
for i in xrange(32):
    active_bin = Bin(sorted(pop[:K*(i+1)]))
    active_bin.label()
    estimates.append(active_bin.auc)

plt.plot(X,estimates, 'b-', label='uniform sample')
plt.show()
