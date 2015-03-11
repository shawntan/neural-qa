import theano
import theano.tensor as T
import numpy as np
import lstm
import cPickle as pickle
from theano_toolkit import utils as U
from theano_toolkit.parameters import Parameters
from theano_toolkit import updates
import model
import vocab
import data_io
import sys

if __name__ == "__main__":
	training_file = sys.argv[1]
	compute_tree_exists = False

	vocab_in,vocab_out = vocab.load("qa1_vocab.pkl")
	if compute_tree_exists:
		inputs,outputs,params,grads = pickle.load(open("compute_tree.pkl"))
	else:
		print "Creating compute tree...",
		P = Parameters()
		story = T.ivector('story')
		idxs  = T.ivector('idxs')
		qstn  = T.ivector('qstn')
		ans_evd = T.iscalar('ans_evd')

		attention = model.build(P,
			word_rep_size = 16,
			stmt_hidden_size = 32,
			diag_hidden_size = 64,
			vocab_size  = len(vocab_in),
			output_size = len(vocab_out),
			map_fun_size = 64
		)
	
		output = attention(story,idxs,qstn)
		print "Done."
		params = P.values()
		cost = -T.log(output[ans_evd])
		print "Calculating gradient expression...",
		grads = T.grad(cost,wrt=params)
		print "Done."
		inputs = [story,idxs,qstn,ans_evd]
		outputs = cost
		pickle.dump(
				(inputs,outputs,params,grads),
				open("compute_tree.pkl","wb"),2
			)

	print "Compiling native...",
	f = theano.function(
			inputs=inputs,
			outputs = cost,
			#updates = [ (p,p-0.1*g) for p,g in zip(params,grads) ]
			updates = updates.adadelta(params,grads) 
		)
	print "Done."

	for epoch in xrange(10):
		group_answers = data_io.group_answers(training_file)
		training_data = data_io.story_question_answer_idx(group_answers,vocab_in,vocab_out)
		training_data = data_io.randomise(training_data)

		for input_data,idxs,question_data,ans_w,ans_evd in training_data:
			print f(input_data,idxs,question_data,ans_evd)
