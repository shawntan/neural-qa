from data_io import *
import model
from theano_toolkit.parameters import Parameters
from theano_toolkit import hinton
from pprint import pprint
import theano.tensor as T
import theano
if __name__ == "__main__":
	group_answers = group_answers(sys.argv[1])
	vocab_in = vocab.load("qa2.pkl")

	vocab_size = len(vocab_in)
	evidence_count = 2

	P = Parameters()
	attention = model.build(P,
		word_rep_size = 50,
		stmt_hidden_size = 100,
		diag_hidden_size = 100,
		vocab_size  = vocab_size,
		output_size = vocab_size,
		map_fun_size = 100,
		evidence_count = evidence_count
	)
	story = T.ivector('story')
	idxs  = T.ivector('idxs')
	qstn  = T.ivector('qstn')
	output_evds,output_ans = attention(story,idxs,qstn)
	answer = theano.function(
			inputs = [story,idxs,qstn],
			outputs = output_evds+[output_ans]
		)

	P.load('tmp.model.pkl')
#	params = pickle.load(open('tmp.model.pkl'))	
#	hinton.plot(params['vocab'])

	training_set = story_question_answer_idx(group_answers,vocab_in)
	rev_map = {}
	for key,val in vocab_in.iteritems(): rev_map[val] = key

	for _ in xrange(2): training_set.next()
	input_data,idxs,question_data,ans_w,ans_evd = training_set.next()

	tokens = [ rev_map[i] for i in input_data ]
	sentences = [ ' '.join(tokens[idxs[i]:idxs[i+1]]) for i in xrange(idxs.shape[0]-1) ]
	pprint(sentences)
	print ' '.join(rev_map[i] for i in question_data)
	for idx in ans_evd:
		print sentences[idx]
	print rev_map[ans_w]
	


	evidence_answer = answer(input_data,idxs,question_data)
	evd_prob = evidence_answer[:evidence_count]
	ans_prob = evidence_answer[-1]	
	
	print "Evidences:"
	for i,e in enumerate(evd_prob): 
		print "predicted",
		hinton.plot(e,max_arr=1)
		correct = np.zeros((e.shape[0],))
		correct[ans_evd[i]] = 1
		print "correct  ",
		hinton.plot(correct)



	print "Answer:"
	print "predicted",
	hinton.plot(ans_prob,max_arr=1)

	correct = np.zeros((ans_prob.shape[0],))
	correct[ans_w] = 1
	print "correct  ",
	hinton.plot(correct,max_arr=1)



