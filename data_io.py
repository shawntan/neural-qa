import re
import sys
import cPickle as pickle
import numpy as np
import vocab
from pprint import pprint
import random
import math
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

def indexify(sentence,vocab):
    return [ vocab[t] for t in sentence ]


def story_question_answer_idx(grouped_answers,vocab):
    for story,question,answer in grouped_answers:
        inputs = [ indexify(s,vocab) for _,s in story ]
        story_data = np.hstack(inputs).astype(dtype=np.int32)
        answer_word      = vocab[answer[0]]
        answer_evidences = answer[1]
        answer_evd_idxs = []
        for answer_evidence in answer_evidences:
            for i,(pos,_) in enumerate(story):
                if pos == answer_evidence:
                    answer_evd_idxs.append(i)
                    break
        idxs = [0]
        for seq in inputs: idxs.append(idxs[-1] + len(seq))
        idxs = np.array(idxs,dtype=np.int32)
        question_data = np.array(indexify(question,vocab),dtype=np.int32)
        
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

def sortify(stream,key=lambda x:x,buffer_size=500):
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

def batch(stream,batch_size=10,criteria=lambda x,y: True):
    def yield_batches(batch):
        random.shuffle(batch)
        sub_batches = int(math.ceil(len(batch) / float(batch_size)))
        for i in xrange(sub_batches):
            yield batch[i * batch_size:(i + 1) * batch_size]

    batch = None
    try:
        while True:
            batch = []
            batch.append(stream.next())
            item = stream.next()
            while criteria(batch[0],item):
                batch.append(item)
                item = stream.next()
            for x in yield_batches(batch): yield x

    except StopIteration:
        for x in yield_batches(batch): yield x

def story_question_answer_idx_(group_answers,vocab):
    for statements,question,answer in group_answers:
        token,evidences = answer
        stmt_count = len(statements)
        stmt_max_length = max(max(len(x) for _,x in statements),len(question))
        stmt_word_idxs = np.empty((stmt_count+1,stmt_max_length),dtype=np.int32)
        stmt_word_idxs[:,:] = -1
        stmt_idx = { stmt_no : stmt_idx
                        for (stmt_no,_),stmt_idx in \
                            zip(statements,xrange(len(statements))) }
        for i,(_,tokens) in enumerate(statements):
            idxs = [vocab[w] for w in tokens]
            stmt_word_idxs[i,-len(idxs):] = idxs

        idxs = [vocab[w] for w in question]
        stmt_word_idxs[-1,-len(idxs):] = idxs
        evidence_idxs = np.array([ stmt_idx[e] for e in evidences ],dtype=np.int32)
        yield stmt_word_idxs, evidence_idxs, vocab[answer[0]]
