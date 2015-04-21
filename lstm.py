import theano
import theano.tensor as T
import numpy as np
import cPickle as pickle

from itertools import izip

from theano_toolkit import utils as U
from theano_toolkit import updates
from theano_toolkit.parameters import Parameters

def build(P,name,input_size,hidden_size):
	name_init_hidden = "init_%s_hidden"%name
	name_init_cell   = "init_%s_cell"%name
	P[name_init_hidden] = np.zeros((hidden_size,),dtype=np.float32)
	P[name_init_cell]   = np.zeros((hidden_size,),dtype=np.float32)
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
	P[name_W_input]  = (0.1 * 2) * (np.random.rand(input_size, hidden_size * 4) - 0.5)
	P[name_W_hidden] = (0.1 * 2) * (np.random.rand(hidden_size,hidden_size * 4) - 0.5)
	P[name_W_cell]   = (0.1 * 2) * (np.random.rand(hidden_size,hidden_size * 3) - 0.5)

	P[name_b] = np.zeros((hidden_size * 4,),dtype=np.float32)
	biases = P[name_b]
	V_if = P[name_W_cell][:,0*hidden_size:2*hidden_size]
	V_o  = P[name_W_cell][:,2*hidden_size:3*hidden_size]

	b_i = biases[0*hidden_size:1*hidden_size]
	b_f = biases[1*hidden_size:2*hidden_size] + 5
	b_c = biases[2*hidden_size:3*hidden_size]
	b_o = biases[3*hidden_size:4*hidden_size]
	def step(x,prev_cell,prev_hid):
		transformed_x = T.dot(x,P[name_W_input])
		x_i = transformed_x[0*hidden_size:1*hidden_size]
		x_f = transformed_x[1*hidden_size:2*hidden_size]
		x_c = transformed_x[2*hidden_size:3*hidden_size]
		x_o = transformed_x[3*hidden_size:4*hidden_size]

		transformed_hid = T.dot(prev_hid,P[name_W_hidden])
		h_i = transformed_hid[0*hidden_size:1*hidden_size]
		h_f = transformed_hid[1*hidden_size:2*hidden_size]
		h_c = transformed_hid[2*hidden_size:3*hidden_size]
		h_o = transformed_hid[3*hidden_size:4*hidden_size]
		
		transformed_cell = T.dot(prev_cell,V_if)
		c_i = transformed_cell[0*hidden_size:1*hidden_size]
		c_f = transformed_cell[1*hidden_size:2*hidden_size]

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
	input_size = 10
	output_size = 20
	P = Parameters()
	lstm_layer = build(P,"test",input_size,output_size)

	cell,_ = lstm_layer(5 * np.random.rand(20,input_size).astype(dtype=np.float32))
	print cell.eval()



