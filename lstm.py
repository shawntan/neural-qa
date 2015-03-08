import theano
import theano.tensor as T
import numpy as np
import cPickle as pickle

from itertools import izip

from theano_toolkit import utils as U
from theano_toolkit import updates
from theano_toolkit.parameters import Parameters

def build(P,name,layer_size,inputs_info,truncate_gradient=-1,bias=True,testing=False):
	gates = ["in","out","forget","cell"]
	weights = {}
	for gate in gates:
		hidden_weight_name = "W_%s_hidden_%s"%(name,gate)
		cell_weight_name   = "W_%s_cell_%s"%(name,gate)
		P[hidden_weight_name] = (0.1 * 2) * (np.random.rand(layer_size,layer_size) - 0.5)
		if gate != "cell":
			P[cell_weight_name] = (0.1 * 2) * (np.random.rand(layer_size,layer_size) - 0.5)
		if bias:
			hidden_bias_name = "b_%s_%s"%(name,gate)
			if gate == "forget":
				P[hidden_bias_name] = 3 + 0.0 * np.random.rand(layer_size)
			else:
				P[hidden_bias_name] = 0.0 * np.random.randn(layer_size)
		for in_name,in_size in inputs_info:
			input_weight_name = "W_%s_%s_%s"%(name,in_name,gate)
			P[input_weight_name] = (0.1 * 2) * (np.random.rand(in_size,layer_size) - 0.5)
			weights[in_name,gate] = P[input_weight_name]

	def transform(x,W):
		if x.type.dtype.startswith('int'): #and len(x.type.broadcastable) == 1:
			return W[x]
		else:
			return T.dot(x,W)
	def lstm_layer(inputs,
			initial_cell = None,
			initial_hidden = None,
			in_activation     = T.nnet.sigmoid,
			out_activation    = T.nnet.sigmoid,
			forget_activation = T.nnet.sigmoid,
			cell_activation   = T.tanh,
		):

		if initial_hidden == None:
			#P["init_%s_hidden"%name] = 0. * U.initial_weights(layer_size)
			#initial_hidden = T.tanh(P["init_%s_hidden"%name])
			initial_hidden = np.zeros((layer_size,),dtype=np.float32)

		if initial_cell == None:
			#P["init_%s_cell"%name] = 0. * U.initial_weights(layer_size)
			#initial_cell = P["init_%s_cell"%name]
			initial_cell = np.zeros((layer_size,),dtype=np.float32)

		U_i = P["W_%s_hidden_in"%name]
		U_o = P["W_%s_hidden_out"%name]
		U_f = P["W_%s_hidden_forget"%name]
		U_c = P["W_%s_hidden_cell"%name]

		V_i = P["W_%s_cell_in"%name]
		V_o = P["W_%s_cell_out"%name]
		V_f = P["W_%s_cell_forget"%name]

		if bias:
			b_i = P["b_%s_in"%name]
			b_o = P["b_%s_out"%name]
			b_f = P["b_%s_forget"%name]
			b_c = P["b_%s_cell"%name]
		else:
			b_i = b_o = b_f = b_c = np.float32(0.)

		X_i = sum(transform(ins,weights[n,"in"])     for ins,(n,_) in zip(inputs,inputs_info))
		X_o = sum(transform(ins,weights[n,"out"])    for ins,(n,_) in zip(inputs,inputs_info))
		X_f = sum(transform(ins,weights[n,"forget"]) for ins,(n,_) in zip(inputs,inputs_info))
		X_c = sum(transform(ins,weights[n,"cell"])   for ins,(n,_) in zip(inputs,inputs_info))

		def step(x_i,x_o,x_f,x_c,prev_cell,prev_hid):
			in_lin     = x_i + T.dot(prev_hid,U_i) + b_i + T.dot(prev_cell,V_i) 
			forget_lin = x_f + T.dot(prev_hid,U_f) + b_f + T.dot(prev_cell,V_f) 
			cell_lin   = x_c + T.dot(prev_hid,U_c) + b_c

			in_gate      = in_activation(in_lin)
			forget_gate  = forget_activation(forget_lin)
			cell_updates = cell_activation(cell_lin)
			cell = forget_gate * prev_cell + in_gate * cell_updates

			out_lin = x_o + T.dot(prev_hid,U_o) + T.dot(cell,V_o) + b_o
			out_gate = out_activation(out_lin)

			hid = out_gate * T.tanh(cell)

			return cell,hid

		if not testing:
			[cell,hidden],_ = theano.scan(
					step,
					sequences    = [X_i,X_o,X_f,X_c],
					outputs_info = [initial_cell,initial_hidden],
					truncate_gradient = truncate_gradient
				)
			return cell,hidden
		else:
			return step(X_i,X_o,X_f,X_c,initial_cell,initial_hidden)

	return lstm_layer
