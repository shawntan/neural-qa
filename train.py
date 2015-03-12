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
def make_functions(inputs,outputs,params,grads):
	acc_grads = [ theano.shared(np.zeros(p.get_value().shape,dtype=np.float32)) for p in params ]
	count = theano.shared(np.float32(0))
	acc_update = [ (a,a+g) for a,g in zip(acc_grads,grads) ] + [ (count,count + 1) ]
	
	avg_acc_grads = [ ag / count for ag in acc_grads ]
	param_update = updates.adadelta(params,avg_acc_grads) + \
					[ (a,np.zeros(p.get_value().shape,dtype=np.float32)) for a,p in zip(acc_grads,params) ] + \
					[ (count,0) ]
	acc = theano.function(
			inputs  = inputs,
			outputs = outputs,
			updates = acc_update
		)
	update = theano.function(inputs=[],updates = param_update)
	return acc,update

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
		ans_lbl = T.iscalar('ans_lbl')

		attention = model.build(P,
			word_rep_size = 16,
			stmt_hidden_size = 32,
			diag_hidden_size = 64,
			vocab_size  = len(vocab_in),
			output_size = len(vocab_out),
			map_fun_size = 64
		)

		output_evd,output_ans = attention(story,idxs,qstn)
		cost = -(T.log(output_evd[ans_evd]) + T.log(output_ans[ans_lbl]))
		print "Done."

		print "Calculating gradient expression...",
		params = P.values()
		grads = T.grad(cost,wrt=params)
		print "Done."

		inputs = [story,idxs,qstn,ans_evd,ans_lbl]
		outputs = cost
		pickle.dump(
				(inputs,outputs,params,grads),
				open("compute_tree.pkl","wb"),2
			)

	print "Compiling native...",
	acc,update = make_functions(inputs,outputs,params,grads)
	print "Done."
	batch_size = 10 
	for epoch in xrange(30):
		group_answers = data_io.group_answers(training_file)
		training_data = data_io.story_question_answer_idx(group_answers,vocab_in,vocab_out)
		training_data = data_io.randomise(training_data)
		
		loss  = 0
		count = 0
		for input_data,idxs,question_data,ans_w,ans_evd in training_data:
			loss  += acc(input_data,idxs,question_data,ans_evd,ans_w)
			count += 1
			if count == batch_size:
				update()
				print loss/count
				loss  = 0
				count = 0

		update()
