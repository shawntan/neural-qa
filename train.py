import sys
import theano
import theano.tensor as T
import numpy as np
import cPickle as pickle
from itertools import islice

import model
import vocab
import data_io
import lstm


from theano_toolkit import utils as U
from theano_toolkit.parameters import Parameters
from theano_toolkit import updates
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

	vocab_in = vocab.load("qa2.pkl")
	vocab_size = len(vocab_in)
	print "Vocab size is:", vocab_size
	entity_size = 20
	evidence_count = 2
	if compute_tree_exists:
		inputs,outputs,params,grads = pickle.load(open("compute_tree.pkl"))
	else:
		print "Creating compute tree...",
		P = Parameters()
		story = T.ivector('story')
		idxs  = T.ivector('idxs')
		qstn  = T.ivector('qstn')
		ans_evds = T.ivector('ans_evds')
		ans_lbl = T.iscalar('ans_lbl')

		attention = model.build(P,
			word_rep_size = 16,
			stmt_hidden_size = 32,
			diag_hidden_size = 64,
			vocab_size  = vocab_size + entity_size,
			output_size = entity_size,
			map_fun_size = 64,
			evidence_count = evidence_count
		)

		output_evds,output_ans = attention(story,idxs,qstn)
		cost = -(
					T.log(output_ans[ans_lbl]) + \
					sum(
						T.log(output_evds[i][ans_evds[i]])
							for i in xrange(evidence_count)
					)
				)
		print "Done."

		print "Calculating gradient expression...",
		params = P.values()
		grads = T.grad(cost,wrt=params)
		print "Done."

		inputs = [story,idxs,qstn,ans_lbl,ans_evds]
		outputs = cost
		pickle.dump(
				(inputs,outputs,params,grads),
				open("compute_tree.pkl","wb"),2
			)

	print "Compiling native...",
	acc,update = make_functions(inputs,outputs,params,grads)
	test = theano.function(
			inputs = [story,idxs,qstn,ans_lbl,ans_evds],
			outputs = [
				T.neq(T.argmax(output_ans),ans_lbl),
			] + [
				T.neq(T.argmax(output_evds[i]),ans_evds[i])
					for i in xrange(evidence_count)
			]
		)


	print "Done."
	batch_size = 10 

	instance_count = sum(1 for _ in data_io.group_answers(training_file))
	train_instance_count = int(0.9 * instance_count)
	print "Total:",instance_count,"Training:",train_instance_count
	best_error = np.inf
	for epoch in xrange(30):
		group_answers = data_io.group_answers(training_file)
		train_group_answers = islice(group_answers,train_instance_count)
		training_data = data_io.story_question_answer_idx(train_group_answers,vocab_in)
		training_data = data_io.randomise(training_data)
		loss  = 0
		count = 0
		for input_data,idxs,question_data,ans_w,ans_evds in training_data:
			loss  += acc(input_data,idxs,question_data,ans_w,ans_evds)
			count += 1
			if count == batch_size:
				update()
				print loss/count
				loss  = 0
				count = 0
		update()

		test_data = data_io.story_question_answer_idx(group_answers,vocab_in)
		errors = sum(
				np.array(test(input_data,idxs,question_data,ans_w,ans_evds),dtype=np.float32)
				for input_data,idxs,question_data,ans_w,ans_evds in test_data
			 )/(instance_count - train_instance_count)
		print errors
		if errors.sum() < best_error:
			P.save('model.pkl')
			best_error = errors.sum()
