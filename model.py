import theano
import theano.tensor as T
import numpy as np
import lstm
from theano_toolkit import utils as U
from theano_toolkit.parameters import Parameters
import cPickle as pickle

def build_hidden_feedforward(P,name,input_size,hidden_size,output_size):
	P["W_%s_hidden"%name] = (0.1 * 2) * (np.random.rand(input_size,hidden_size) - 0.5)
	P["b_%s_hidden"%name] = np.zeros((hidden_size,),dtype=np.float32)
	P["W_%s_output"%name] = np.zeros((hidden_size,output_size),dtype=np.float32)
	P["b_%s_output"%name] = np.zeros((output_size,),dtype=np.float32)
	W_hidden = P["W_%s_hidden"%name]
	b_hidden = P["b_%s_hidden"%name]
	W_output = P["W_%s_output"%name]
	b_output = P["b_%s_output"%name]
	def feedforward(X):
		hidden = T.tanh(T.dot(X,W_hidden) + b_hidden)
		key = T.dot(hidden,W_output) + b_output 
		return key
	return feedforward


def build_stmt_encoder(P,input_size,hidden_size):
	lstm_layer = lstm.build(P,"stmt",input_size,hidden_size)
	def encode_stmt(X):
		cells,_ = lstm_layer(X)
		return cells[-1]
	return encode_stmt

def build_diag_encoder(P,stmt_size,hidden_size,encode_stmt):
	lstm_layer = lstm.build(P,"diag",stmt_size,hidden_size)
	def encode_diag(X,idxs):
		cells,_ = lstm_layer(
			T.arange(idxs.shape[0]-1),
			row_transform = (lambda i:encode_stmt(X[idxs[i]:idxs[i+1]]))
		)
		return cells
	return encode_diag


def build(P,word_rep_size,stmt_hidden_size,diag_hidden_size,vocab_size,output_size):
	P.W_vocab = (0.1 * 2) * (np.random.rand(vocab_size,word_rep_size) - 0.5)
	encode_stmt = build_stmt_encoder(P,word_rep_size,stmt_hidden_size)
	encode_diag = build_diag_encoder(P,stmt_hidden_size,diag_hidden_size,encode_stmt)
	stmt2key = build_hidden_feedforward(P,"stmt2key",stmt_hidden_size,64,diag_hidden_size)
	diag2output = build_hidden_feedforward(P,"stmt2output",diag_hidden_size,64,output_size)
	def qa(story,idxs,qstn):
		word_feats = P.W_vocab[story]
		qn_word_feats = P.W_vocab[qstn]
		cum_stmt_vectors = encode_diag(word_feats,idxs)
		qn_vector = encode_stmt(qn_word_feats)
		key = stmt2key(qn_vector)
		stmt_attention = U.vector_softmax(T.dot(key,cum_stmt_vectors.T))
		selected_vector = T.dot(stmt_attention,cum_stmt_vectors)
		output = U.vector_softmax(diag2output(selected_vector))
		return stmt_attention,output
	return qa

if __name__ == "__main__":
	import vocab
	import data_io
	import sys
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
		attention = build(P,10,20,40,len(vocab_in),len(vocab_out))
		output_evd,output_ans = attention(story,idxs,qstn)
		print "Done."
		params = P.values()
		cost = -(T.log(output_evd[ans_evd]) + T.log(output_ans[ans_lbl]))
		print "Calculating gradient expression...",
		grads = T.grad(cost,wrt=params)
		print "Done."
		inputs = [story,idxs,qstn,ans_evd,ans_lbl]
		outputs = cost
#		pickle.dump(
#				(inputs,outputs,params,grads),
#				open("compute_tree.pkl","wb"),2
#			)

	print "Compiling native...",
	f = theano.function(
			inputs=inputs,
			outputs=outputs,
			updates = [ (p,p-0.01*g) for p,g in zip(params,grads) ]
		)
	print "Done."
	for epoch in xrange(10):
		group_answers = data_io.group_answers(sys.argv[1])
		training_data = data_io.story_question_answer_idx(group_answers,vocab_in,vocab_out)
		
		for input_data,idxs,question_data,ans_w,ans_evd in training_data:
			print f(input_data,idxs,question_data,ans_evd,ans_w)
