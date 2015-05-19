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
def combined_probs(prob_dists,idxs,pos=0):
	if len(prob_dists) > 0:
		total_prob = 0
		for i in xrange(len(prob_dists)):
			total_prob += prob_dists[i][idxs[pos]] * \
					combined_probs(
						prob_dists[:i] + prob_dists[i+1:],
						idxs,pos=pos+1
					)
		return total_prob
	else:
		return 1

def ordered_probs(prob_dists,idxs):
	result = 1
	for i,p in enumerate(prob_dists):
		result *= p[idxs[i]]
	return result



def make_functions(inputs,outputs,params,grads,lr):
	shapes = [ p.get_value().shape for p in params ]
	acc_grads = [ theano.shared(np.zeros(s,dtype=np.float32)) for s in shapes ]
	count = theano.shared(np.float32(0))
	acc_update = [ (a,a+g) for a,g in zip(acc_grads,grads) ] + [ (count,count + 1) ]

	deltas = acc_grads
#	deltas      = [ ag / count for ag in acc_grads ]
	grads_norms = [ T.sqrt(T.sum(g**2)) for g in deltas ]
	deltas      = [ T.switch(T.gt(n,30),30*g/n,g)
					for n,g in zip(grads_norms,deltas) ]
	
	param_update = [ (p, p - lr * g) for p,g in zip(params,deltas) ]
#	param_update = updates.adadelta(params,deltas)
	
	clear_update = [ 
			(a,np.zeros(s,dtype=np.float32)) 
			for a,s in zip(acc_grads,shapes) 
		] + [ (count,0) ]
	acc = theano.function(
			inputs  = inputs,
			outputs = outputs,
			updates = acc_update,
			on_unused_input='warn'
		)
	update = theano.function(
			inputs=[lr],
			updates = param_update + clear_update,
			on_unused_input='warn'
		)
	return acc,update

if __name__ == "__main__":
	training_file = sys.argv[1]
	compute_tree_exists = False

	vocab_in = vocab.load("qa2.pkl")
	vocab_size = len(vocab_in)
	print "Vocab size is:", vocab_size
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
			word_rep_size = 50,
			stmt_hidden_size = 100,
			diag_hidden_size = 100,
			vocab_size  = vocab_size,
			output_size = vocab_size,
			map_fun_size = 100,
			evidence_count = evidence_count
		)

		output_evds,output_ans = attention(story,idxs,qstn)
		cost = -T.log(output_ans[ans_lbl]) 
		#cost += 1e-5 * sum(T.sum(w**2) for w in P.values() )
		#cost += -T.log(ordered_probs(output_evds,ans_evds)) 
		print "Done."
		print "Parameter count:", P.parameter_count()

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
	lr = T.fscalar('lr')
	acc,update = make_functions(inputs,outputs,params,grads,lr)
	test = theano.function(
			inputs = [story,idxs,qstn,ans_lbl,ans_evds],
			outputs =  1 - T.eq(T.argmax(output_ans),ans_lbl),
			on_unused_input='warn'
		)
	"""
		1 - T.eq(T.argmax(output_ans),ans_lbl) * T.prod(T.eq(
			T.sort(T.argmax(T.stack(*output_evds),axis=1)),
			T.sort(ans_evds)
		)),
	"""


	print "Done."

	instance_count = 0
	for _ in data_io.group_answers(training_file):
		instance_count += 1

	test_instance_count = int(0.1 * instance_count)
	print "Total:",instance_count,"Testing:",test_instance_count
	best_error = 1. 

	#P.load('model.pkl')

	batch_size = 16
	length_limit = 13
	learning_rate = 0.01
	epoch = 1
	while True:
		group_answers = data_io.group_answers(training_file)
		test_group_answers = islice(group_answers,test_instance_count)
		test_data = data_io.story_question_answer_idx(
						test_group_answers,
						vocab_in
					)
		errors = sum(
				np.array(
					test(input_data,idxs,question_data,ans_w,ans_evds),
					dtype=np.float32
				)
				for input_data,idxs,question_data,ans_w,ans_evds in test_data
			 )/test_instance_count
		print "Error rate:",errors
		print "Starting epoch ",epoch
		if errors <= best_error:
			P.save('model.pkl')
			best_error = errors
			length_limit += 2
		else:
#			learning_rate = learning_rate / 2
#			batch_size = max(1,batch_size//2)
#			print "Learning rate:",learning_rate
			P.save('tmp.model.pkl')
		buffer_size = 256 / batch_size

		train_group_answers = data_io.randomise(group_answers)
		training_data = data_io.story_question_answer_idx(train_group_answers,vocab_in)
		training_data = ( x for x in training_data if x[1].shape[0] <= length_limit )
		training_data = data_io.sortify(training_data,key=lambda x:x[1].shape[0])
		batched_training_data = data_io.batch(training_data,batch_size=batch_size)
		batched_training_data = data_io.randomise(batched_training_data,buffer_size=buffer_size)

		for batch in batched_training_data:
			loss  = 0
			count = 0
			for input_data,idxs,question_data,ans_w,ans_evds in batch:
					print idxs.shape[0],
					loss  += acc(input_data,idxs,question_data,ans_w,ans_evds)
					count += 1
			update(learning_rate)
			print loss/count
		epoch += 1
		if epoch % 15 == 0:
			learning_rate = learning_rate / 2
		if epoch % 30 == 0:
			batch_size = max(batch_size // 2,1)
			



