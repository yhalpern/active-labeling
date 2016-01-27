import random
import h5py

'''
create a dataset with the monotonic property.
P(y=1) increases monotically with yhat.
'''
N = 100000
random.seed(123)


def monotonic_f(x):
    return 0.5*x + 0.3

def draw_yhat():
    return random.betavariate(1,3)

def draw_y(yhat):
    p = monotonic_f(yhat)
    return p > random.random()

def draw_pair():
    yhat = draw_yhat()
    y = draw_y(yhat)
    return y,yhat

Y = []
for _ in xrange(N):
    y,yhat = draw_pair()
    Y.append((yhat,y))

Y.sort()
Yhat,Y = zip(*Y)

f = h5py.File('data.h5py', 'w')
f.create_dataset('Y', data=Y)
f.create_dataset('Yhat', data=Yhat)
f.close()
