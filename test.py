import theano
import theano.tensor as T
import numpy as np
import cPickle as pickle

from theano_toolkit import utils as U
from theano_toolkit.parameters import Parameters

grad_exists = True

if grad_exists:
	print "Loading graph from save file."
	X,y,dX = pickle.load(open('test_grad.pkl'))
else:
	print "Recreating..."
	X = theano.shared(1000*np.random.rand(1))
	x = X[0]
	y = 5*(x**2) - 10*x + 21
	dX = T.grad(y,wrt=X)
	pickle.dump((X,y,dX),open('test_grad.pkl','wb'),2)

X.set_value(np.array([20]))
f = theano.function(
		inputs = [],
		outputs = y,
		updates = [ (X,X - 0.01 * dX) ]
	)

for _ in xrange(100): print f()
