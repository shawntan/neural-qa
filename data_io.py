import re
import sys
import cPickle as pickle
import numpy as np
from pprint import pprint
p1 = re.compile('([^ ])([\?\.\,\!\%])')
p2 = re.compile('([\?\.\,\!\%])([^ ])')

def clean_string(text):
	text = text.strip()
	text = text.lower()
	text = p1.sub(r"\1 \2",text)
	text = p2.sub(r"\1 \2",text)
	return text

def process_lines(filename):
	for line in open(filename,'r'):
		line = line.strip()
		parts = line.split('\t')
		input_sentence = parts[0].split(' ',1)
		input_sentence[0] = int(input_sentence[0])
		input_sentence[1] = clean_string(input_sentence[1].strip()).split()
		if len(parts) < 3:
			response_sentence = None
		else:
			response_sentence = parts[-2:]
			response_sentence[0] = clean_string(response_sentence[0])
			response_sentence[1] = map(int,response_sentence[1].split())
		yield (input_sentence,response_sentence)
	

def sessions(lines):
	session = []
	prev_id = 0
	for input_s,output_s in lines:
		if input_s[0] < prev_id:
			yield session
			session = []
		prev_id = input_s[0]
		session.append((input_s,output_s))


def group_answers(filename):
	for session in sessions(process_lines(filename)):
		prev_inputs = []
		for input_sentence,response_sentence in session:
			if response_sentence == None:
				prev_inputs.append(input_sentence)
			else:
				yield (prev_inputs,input_sentence[1],response_sentence)

def indexify(tokens,vocab):
	tokens = map(vocab.get,tokens)
	return tokens


def story_question_answer_idx(grouped_answers,vocabfile):
	vocab = pickle.load(open(vocabfile,'rb'))
	vocab_in  = { i:k for k,i in enumerate(vocab['input_vocab']) }
	vocab_out = { i:k for k,i in enumerate(vocab['output_vocab']) }
	for story,question,answer in grouped_answers:
		inputs = [ indexify(tokens,vocab_in) for _,tokens in story ]
		story_data = np.hstack(inputs)

		idxs = [0]
		for seq in inputs: idxs.append(idxs[-1] + len(seq))
		idxs = np.array(idxs)
	
		question_data = np.array(indexify(question,vocab_in))
		
		answer_word = vocab_out[answer[0]]
		answer_evidence = answer[1][0]

		yield story_data,idxs,question_data,answer_word,answer_evidence



if __name__ == "__main__":
	group_answers = group_answers(sys.argv[1])
	for input_data,idxs,question_data,ans_w,ans_evd in story_question_answer_idx(group_answers,sys.argv[2]):
		print input_data
		print idxs
		print question_data
		print ans_w,ans_evd
		print
