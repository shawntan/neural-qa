import sys
import theano
import theano.tensor as T
import numpy as np
import cPickle as pickle
from itertools import islice

import model
import vocab
import data_io
from pprint import pprint

from theano_toolkit import utils as U
from theano_toolkit.parameters import Parameters
from theano_toolkit import updates

def make_functions(inputs,outputs,params,grads,lr):
    shapes = [ p.get_value().shape for p in params ]
    acc_grads = [ theano.shared(np.zeros(s,dtype=np.float32)) for s in shapes ]
    count = theano.shared(np.float32(0))
    acc_update = [ (a,a+g) for a,g in zip(acc_grads,grads) ] + [ (count,count + 1.) ]
    deltas     = [ ag / count for ag in acc_grads ]
    param_update = updates.adadelta(params,deltas,learning_rate=lr,delta_preprocess=updates.clip(1.)) # ,learning_rate=lr,rho=np.float32(0.95)

    clear_update = [ 
            (a,np.zeros(s,dtype=np.float32)) 
            for a,s in zip(acc_grads,shapes) 
            ] + [ (count,0) ]
    acc = theano.function(
            inputs  = inputs,
            outputs = [outputs,T.eq(T.argmax(output_ans),ans_lbl)],
            updates = acc_update,
            on_unused_input='warn',
#            mode=theano.compile.MonitorMode(post_func=detect_nan)
        )
    update = theano.function(
            inputs=[lr],
            updates = param_update + clear_update,
            outputs = [ T.sqrt(T.sum(T.sqr(w))) for w in deltas ],
            on_unused_input='warn',
#            mode=theano.compile.MonitorMode(post_func=detect_nan)
        )
    return acc,update

if __name__ == "__main__":
    training_file = sys.argv[1]
    compute_tree_exists = False

    vocab_in = vocab.load("qa2.pkl")
    vocab_size = len(vocab_in)
    print "Vocab size is:", vocab_size
    evidence_count = 2
    print "Creating compute tree...",
    P = Parameters()
    story = T.imatrix('story')
    ans_evds = T.ivector('ans_evds')
    ans_lbl = T.iscalar('ans_lbl')

    attention = model.build(P,
            vocab_size=vocab_size + 1,
            word_rep_size=vocab_size,
            sentence_rep_size=128,
            output_size=vocab_size,
            evidence_count=evidence_count
        )

    output_ans,output_evds = attention(story)

    cross_entropy = -T.log(output_ans[ans_lbl])  \
            + -T.log(output_evds[0][ans_evds[0]]) \
            + -T.log(output_evds[1][ans_evds[1]]) 
    #cost += -T.log(ordered_probs(output_evds,ans_e.vds)) 
    print "Done."
    print "Parameter count:", P.parameter_count()

    print "Calculating gradient expression...",
    params = P.values()
    cost = cross_entropy #+ 1e-5 * sum(T.sum(T.sqr(w)) for w in params)
    grads = T.grad(cost,wrt=params)
    print "Done."

    inputs = [story,ans_lbl,ans_evds]
    outputs = cross_entropy

    print "Compiling native...",
    lr = T.fscalar('lr')
    acc,update = make_functions(inputs,outputs,params,grads,lr)
    test = theano.function(
            inputs = [story,ans_lbl],
            outputs =  1 - T.eq(T.argmax(output_ans),ans_lbl),
            on_unused_input='warn'
            )
    print "Done."

    instance_count = 0
    for _ in data_io.group_answers(training_file):
        instance_count += 1

    test_instance_count = int(0.1 * instance_count)
    print "Total:",instance_count,"Testing:",test_instance_count
    best_error = 1.

    #P.load('model.pkl')

    batch_size = 32
    length_limit = np.inf
    learning_rate = 1e-6
    epoch = 1
    while True:
        group_answers = data_io.group_answers(training_file)
        test_group_answers = islice(group_answers,test_instance_count)
        test_data = data_io.story_question_answer_idx_(
                        test_group_answers,
                        vocab_in
                    )
        test_data = ( x for x in test_data if x[1].shape[0] <= length_limit )
        tests = [ np.array(
                    test(input_data,ans_w),
                    dtype=np.float32
                )
                for input_data,ans_evds,ans_w in test_data ]
        errors = sum(tests)/len(tests)
        print "Error rate:",errors
        print "Starting epoch ",epoch
        if errors < best_error * 0.9 :
            P.save('model.pkl')
            print "Wrote model."
            best_error = errors
            length_limit += 2
        else:
            P.save('tmp.model.pkl')

        buffer_size = 256 / batch_size

        train_group_answers = data_io.randomise(group_answers)
        training_data = data_io.story_question_answer_idx_(train_group_answers,vocab_in)
        training_data = data_io.sortify(training_data,key=lambda x:x[0].shape[0])
        batched_training_data = data_io.batch(
                training_data,
                batch_size=batch_size,
                criteria=lambda x,x_:abs(x[0].shape[0] - x_[0].shape[0]) <= 2
            )
        batched_training_data = data_io.randomise(batched_training_data,buffer_size=buffer_size)
        
        group_count = 0
        for batch in batched_training_data:
            loss  = 0
            count = 0
            for input_data,ans_evds,ans_w in batch:
                print input_data.shape[0],
                curr_loss  = np.array(acc(input_data,ans_w,ans_evds))
                if np.isnan(curr_loss).any():
                    print curr_loss 
                    exit()
                loss  += curr_loss
                count += 1
                group_count += 1
            change = update(learning_rate)
            print
            pprint({ p.name:c for p,c in zip(params,change) })
            print loss/count

        print "Seen",group_count,"groups"
        epoch += 1

