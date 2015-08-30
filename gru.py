import theano
import theano.tensor as T
import numpy as np
import cPickle as pickle

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


def build(P, name, input_size, hidden_size, truncate_gradient=-1):
    P["init_%s_hidden" % name] = np.zeros((hidden_size,))
    _step, transform = _build_step(P, name, input_size, hidden_size, batched=True)
    def gru_layer(X):
        init_hidden = P["init_%s_hidden" % name]
        init_hidden = T.tanh(init_hidden)
        init_hidden_batch = T.alloc(init_hidden, X.shape[1], hidden_size)
        hidden, _ = theano.scan(
            _step,
            sequences=[transform(X)],
            outputs_info=[init_hidden_batch],
            truncate_gradient=truncate_gradient
        )
        return hidden
    return gru_layer


def build_step(P, name, input_size, hidden_size):
    _step, _ = _build_step(P, name, input_size, hidden_size, batched=False)

    def step(x, prev_cell, prev_hid):
        x = x.dimshuffle('x', 0)
        prev_cell = prev_cell.dimshuffle('x', 0)
        prev_hid = prev_hid.dimshuffle('x', 0)
        cell, hid = _step(x, prev_cell, prev_hid)
        return cell[0], hid[0]
    return step


def _build_step(P, name, input_size, hidden_size, batched):
    name_W_input = "W_%s_input" % name
    name_W_hidden = "W_%s_hidden" % name
    name_W_cell = "W_%s_cell" % name
    name_b = "b_%s" % name
    P[name_W_input]  = 0.01 * np.random.rand(input_size,  hidden_size * 3)
    transition_weights = np.empty((hidden_size, hidden_size * 3),dtype=np.float32)
    transition_weights[:,0 * hidden_size:1 * hidden_size] = orthogonal_init(hidden_size,hidden_size)
    transition_weights[:,1 * hidden_size:2 * hidden_size] = orthogonal_init(hidden_size,hidden_size)
    transition_weights[:,2 * hidden_size:3 * hidden_size] = orthogonal_init(hidden_size,hidden_size)
    P[name_W_hidden] = transition_weights

    bias_init = np.zeros((3, hidden_size), dtype=np.float32)
    bias_init[1] = 2.5
    P[name_b] = bias_init
    biases = P[name_b]

    b_z = biases[0]
    b_r = biases[1]
    b_h = biases[2]

    def _transform(x):
        """
        Input dimensions:  time x batch_size x input_size
        Output dimensions: time x batch_size x 4 x hidden_size
        """
        x_flatten = x.reshape((x.shape[0] * x.shape[1], input_size)) # time * batch_size x input_size
        _transformed_x = T.dot(x_flatten, P[name_W_input])           # time * batch_size x 3 * hidden_size
        _transformed_x = _transformed_x\
            .reshape((x.shape[0], x.shape[1], 3, hidden_size))       # time x batch_size x 3 * hidden_size
        return _transformed_x

    def _step(x, prev_h):
        if batched:
            transformed_x = x
        else:
            transformed_x = T.dot(x, P[name_W_input]).reshape((1, 4, hidden_size))

        # batch_size x hidden_size

        batch_size = transformed_x.shape[0]
        transformed_x = transformed_x.reshape((batch_size, 3, hidden_size))                     # batch_size x 3 x hidden_size
        transformed_hid = T.dot(prev_h, P[name_W_hidden]).reshape((batch_size, 3, hidden_size)) # batch_size x 3 x hidden_size

        transformed_x_ = transformed_x.dimshuffle(1, 0, 2)
        x_z = transformed_x_[0]
        x_r = transformed_x_[1]
        x_h = transformed_x_[2]   # batch_size x hidden_size

        transformed_hid_ = transformed_hid.dimshuffle(1, 0, 2)
        h_z = transformed_hid_[0]
        h_r = transformed_hid_[1]
        h_h = transformed_hid_[2] # batch_size x hidden_size


        z_lin = x_z + h_z + b_z
        r_lin = x_r + h_r + b_r

        r = T.nnet.sigmoid(r_lin)
        z = T.nnet.sigmoid(z_lin)

        h_lin = x_h + r * h_h + b_h
        curr_h = T.tanh(h_lin)

        h = z * prev_h + (1 - z) * curr_h

        return h

    return _step, _transform
