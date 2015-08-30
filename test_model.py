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
            vocab_size=vocab_size + 1,
            word_rep_size=vocab_size,
            sentence_rep_size=128,
            output_size=vocab_size,
            evidence_count=evidence_count
        )

    story = T.imatrix('story')
    output_ans,output_evds = attention(story)
    answer = theano.function(
            inputs = [story],
            outputs = output_evds+[output_ans]
        )

    P.load('tmp.model.pkl')
    #P.load(open('tmp.model.pkl'))    
#    hinton.plot(params['vocab'])

    training_set = story_question_answer_idx_(group_answers,vocab_in)
    rev_map = {}
    for key,val in vocab_in.iteritems(): rev_map[val] = key

    rev_map[-1] = "<unk>"
    import random
    
    for _ in xrange(3): training_set.next()
    input_data,ans_evds,ans_w = training_set.next()
    sentences = [ ' '.join(rev_map[input_data[i,j]] 
                    for j in xrange(input_data.shape[1]))
                        for i in xrange(input_data.shape[0]) ]
    pprint(sentences)
    evidence_answer = answer(input_data)
    evd_prob = evidence_answer[:evidence_count]
    ans_prob = evidence_answer[-1]
    
    print "Evidences:"
    for i,e in enumerate(evd_prob): 
        print e
        print "predicted",
        hinton.plot(e,max_arr=1)
        correct = np.zeros((e.shape[0],))
        correct[ans_evds[i]] = 1
        print "correct  ",
        hinton.plot(correct)

    print "Answer:"
    print "predicted",
    hinton.plot(ans_prob,max_arr=1)

    correct = np.zeros((ans_prob.shape[0],))
    correct[ans_w] = 1
    print "correct  ",
    hinton.plot(correct,max_arr=1)



