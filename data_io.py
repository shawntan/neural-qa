import re
import sys
import cPickle as pickle
import numpy as np
import vocab
from pprint import pprint
import random
p1 = re.compile('([^ ])([\?\.\,\!\%])')
p2 = re.compile('([\?\.\,\!\%])([^ ])')

def clean_string(text,lower):
	text = text.strip()
	if lower: text = text.lower()
	text = p1.sub(r"\1 \2",text)
	text = p2.sub(r"\1 \2",text)
	return text

def process_lines(filename,lower=True):
	for line in open(filename,'r'):
		line = line.strip()
		parts = line.split('\t')
		input_sentence = parts[0].split(' ',1)
		input_sentence[0] = int(input_sentence[0])
		input_sentence[1] = clean_string(input_sentence[1].strip(),lower).split()
		if len(parts) < 3:
			response_sentence = None
		else:
			response_sentence = parts[-2:]
			response_sentence[0] = clean_string(response_sentence[0],lower)
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

def indexify(tokens,vocab,entity_map,no_create=False):
	result = []
	for t in tokens:
		if t in vocab:
			result.append(vocab[t])
		else:
			if t not in entity_map:
				assert(not no_create)
				entity_map[t] = len(vocab) + len(entity_map)
			result.append(entity_map[t])
	return result


def story_question_answer_idx(grouped_answers,vocab_in):
	for story,question,answer in grouped_answers:
		
		entity_map = {}
		inputs = [ indexify(tokens,vocab_in,entity_map) for _,tokens in story ]
		story_data = np.hstack(inputs).astype(dtype=np.int32)

		answer_word     = entity_map[answer[0]] - len(vocab_in)
		answer_evidences = answer[1]
		assert(answer_word >= 0)
		answer_evd_idxs = []
		for answer_evidence in answer_evidences:
			for i,(pos,_) in enumerate(story):
				if pos == answer_evidence:
					answer_evd_idxs.append(i)
					break
	
		idxs = [0]
		for seq in inputs: idxs.append(idxs[-1] + len(seq))
		idxs = np.array(idxs,dtype=np.int32)
		question_data = np.array(indexify(question,vocab_in,entity_map),dtype=np.int32)
		
		yield story_data,idxs,question_data,answer_word,answer_evd_idxs

def randomise(stream,buffer_size=100):
	buf = buffer_size * [None]
	ptr = 0
	for item in stream:
		buf[ptr] = item
		ptr += 1
		if ptr == buffer_size:
			random.shuffle(buf)
			for x in buf: yield x
			ptr = 0
	buf = buf[:ptr]
	random.shuffle(buf)
	for x in buf: yield x

def sortify(stream,key,buffer_size=100):
	buf = buffer_size * [None]
	ptr = 0
	for item in stream:
		buf[ptr] = item
		ptr += 1
		if ptr == buffer_size:
			buf.sort(key=key)
			for x in buf: yield x
			ptr = 0
	buf = buf[:ptr]
	buf.sort(key=key)
	for x in buf: yield x


if __name__ == "__main__":
	group_answers = group_answers(sys.argv[1])

	vocab_in = vocab.load("qa2.pkl")
	training_set = story_question_answer_idx(group_answers,vocab_in)

	for input_data,idxs,question_data,ans_w,ans_evd in training_set:
		print input_data
		print idxs
		print question_data
		print ans_w,ans_evd
		print
