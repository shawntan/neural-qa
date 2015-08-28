import theano
import theano.tensor as T
import numpy as np
import cPickle as pickle

from itertools import izip

from theano_toolkit import utils as U
from theano_toolkit import updates
from theano_toolkit.parameters import Parameters

def orthogonal_init(*dimensions):
	flat_dimensions = (dimensions[0], np.prod(dimensions[1:]))
	a = np.random.randn(*flat_dimensions)
	u,_,v = np.linalg.svd(a, full_matrices=False)
	q = u if u.shape == flat_dimensions else v # pick the one with the correct shape
	q = q.reshape(dimensions)
	return q

def build(P,name,input_size,hidden_size):
	name_init_hidden = "init_%s_hidden"%name
	name_init_cell   = "init_%s_cell"%name
	P[name_init_hidden] = 0.0 * np.random.randn(hidden_size)
	P[name_init_cell]   = 0.0 * np.random.randn(hidden_size)

	step = build_step(P,name,input_size,hidden_size)

	def lstm_layer(X,row_transform=lambda x:x):
		init_hidden = T.tanh(P[name_init_hidden])
		init_cell   = P[name_init_cell]

		def _step(x,prev_cell,prev_hid):
			if row_transform != None: x = row_transform(x)
			return step(x,prev_cell,prev_hid)
		[cell,hidden],_ = theano.scan(
				_step,
				sequences    = [X],
				outputs_info = [init_cell,init_hidden],
			)
		return cell,hidden
	return lstm_layer

def build_step(P,name,input_size,hidden_size):
	name_W_input  = "W_%s_input"%name
	name_W_hidden = "W_%s_hidden"%name
	name_W_cell   = "W_%s_cell"%name
	name_b        = "b_%s"%name
	P[name_W_input]  = orthogonal_init(input_size,  hidden_size * 4)
	P[name_W_hidden] = orthogonal_init(hidden_size, hidden_size * 4)
	P[name_W_cell]   = orthogonal_init(hidden_size, hidden_size * 3)

#	(0.1 * 2) * (np.random.rand(input_size, hidden_size * 4) - 0.5)
#	(0.1 * 2) * (np.random.rand(hidden_size,hidden_size * 4) - 0.5)
#	(0.1 * 2) * (np.random.rand(hidden_size,hidden_size * 3) - 0.5)
	
	bias_init = np.zeros((4,hidden_size),dtype=np.float32)
	bias_init[1] = 2.5
	P[name_b] = bias_init
	biases = P[name_b]

	V_if = P[name_W_cell][:,0*hidden_size:2*hidden_size]
	V_o  = P[name_W_cell][:,2*hidden_size:3*hidden_size]

	b_i = biases[0]
	b_f = biases[1]
	b_c = biases[2]
	b_o = biases[3]
	def step(x,prev_cell,prev_hid):
		transformed_x = T.dot(x,P[name_W_input]).reshape((4,hidden_size))
		x_i = transformed_x[0]
		x_f = transformed_x[1]
		x_c = transformed_x[2]
		x_o = transformed_x[3]

		transformed_hid = T.dot(prev_hid,P[name_W_hidden]).reshape((4,hidden_size))
		h_i = transformed_hid[0]
		h_f = transformed_hid[1]
		h_c = transformed_hid[2]
		h_o = transformed_hid[3]
		
		transformed_cell = T.dot(prev_cell,V_if).reshape((2,hidden_size))
		c_i = transformed_cell[0]
		c_f = transformed_cell[1]

		in_lin     = x_i + h_i + b_i + c_i
		forget_lin = x_f + h_f + b_f + c_f
		cell_lin   = x_c + h_c + b_c

		in_gate      = T.nnet.sigmoid(in_lin)
		forget_gate  = T.nnet.sigmoid(forget_lin)
		cell_updates = T.tanh(cell_lin)

		cell = forget_gate * prev_cell + in_gate * cell_updates

		out_lin = x_o + h_o + b_o + T.dot(cell,V_o)
		out_gate = T.nnet.sigmoid(out_lin)

		hid = out_gate * T.tanh(cell)
		return cell,hid
	return step


if __name__ == "__main__":
	P = Parameters()
	X = T.ivector('X')
	P.V = np.zeros((8,8),dtype=np.int32)

	X_rep = P.V[X]
	P.W_output = np.zeros((15,8),dtype=np.int32)
	lstm_layer = build(P,
			name = "test",
			input_size = 8,
			hidden_size =15 
		)

	_,hidden = lstm_layer(X_rep)
	output = T.nnet.softmax(T.dot(hidden,P.W_output))
	delay = 5
	label = X[:-delay]
	predicted = output[delay:]

	cost = -T.sum(T.log(predicted[T.arange(predicted.shape[0]),label]))
	params = P.values()
	gradients = T.grad(cost,wrt=params)


	update_methods = {
			'standard': [ (p, p - 0.001 * g) for p,g in zip(params,gradients) ],
#			'rmsprop' : updates.rmsprop(params,gradients),
#			'adadelta': updates.rmsprop(params,gradients),
		}
	P.save('init.pkl')
	for update_method in update_methods:
		print "Using update method:",update_method
		with open('train.%s.smart_init.log'%update_method,'w') as log:

			train = theano.function(
					inputs = [X],
					outputs = cost,
					updates = update_methods[update_method],
				)

			P.load('init.pkl')

			while True:
				cost_val = train(np.random.randint(0,8,size=20).astype(np.int32))
				log.write("%0.5f\n"%cost_val)
				print cost_val
				if cost_val < 0.01:
					break
		P.save('lstm.%s.smart_init.pkl'%update_method)

			


	




