import theano
import theano.tensor as T
import numpy as np
import cPickle as pickle
from theano_toolkit import utils as U
from theano_toolkit.parameters import Parameters
import cPickle as pickle
import gru
def random_init(*dimensions):
    return np.random.randn(*dimensions)

def zeros_init(*dimensions):
    return np.zeros(dimensions,dtype=np.float32)

def orthogonal_init(*dimensions):
    flat_dimensions = (dimensions[0], np.prod(dimensions[1:]))
    a = np.random.randn(*flat_dimensions)
    u,_,v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_dimensions else v # pick the one with the correct shape
    q = q.reshape(dimensions)
    return q


def build_stmt_encoder(P,input_size,hidden_size):
    gru_layer = gru.build(
            P,
            name="gru",
            input_size=input_size,
            hidden_size=hidden_size
        )

    def encode_stmts(X):
        X = X.dimshuffle(1,0,2)
        return gru_layer(X)[-1]
    return encode_stmts

def build_lookup(P,input_size,key_size,hidden_size=128):
    P.W_lookup_input_hidden = 0.1 * random_init(input_size,hidden_size)
    P.W_lookup_key_hidden = 0.1 * random_init(key_size,hidden_size)
    P.b_lookup_hidden = zeros_init(hidden_size)
    P.W_lookup = 0.01 * random_init(hidden_size)
    P.W_before_after = 0.00 * random_init(key_size)
    P.b_before_after = 0.0
    P.time_factor = 0.
    def prepare_lookup(sentence_reps):
        _hidden = T.dot(sentence_reps,P.W_lookup_input_hidden)
        time = T.arange(sentence_reps.shape[0])
        def lookup(key,prev_attn):
            sim_score = T.dot(T.tanh(
                    T.dot(key,P.W_lookup_key_hidden) +\
                    _hidden + P.b_lookup_hidden
                ),P.W_lookup)
            before_after = T.dot(key,P.W_before_after) + P.b_before_after
            prev_location = T.dot(prev_attn,time)
            before_after_prob = T.nnet.sigmoid(before_after * (time - prev_location))

            score = T.exp(sim_score + P.time_factor * time) *\
                    (1 - prev_attn) * before_after_prob
            probs = score / T.sum(score)
            return probs
        return lookup
    return prepare_lookup

def build_reasoner(P,input_size,hidden_size):
    P.W_reasoner_input_hidden = 0.1 * random_init(input_size,hidden_size)
    P.W_reasoner_hidden_hidden = orthogonal_init(hidden_size,hidden_size)
    P.b_reasoner_hidden = zeros_init(hidden_size)
    def reason(state,evidence):
        return T.tanh(
                T.dot(state,P.W_reasoner_hidden_hidden) +\
                T.dot(evidence,P.W_reasoner_input_hidden) +\
                P.b_reasoner_hidden
            )
    return reason


def build(P,
        vocab_size,
        word_rep_size,
        sentence_rep_size,
        output_size,
        evidence_count
    ):

    P.vocab = zeros_init(vocab_size,word_rep_size)

    encode_stmts = build_stmt_encoder(
            P,
            input_size=word_rep_size,
            hidden_size=sentence_rep_size
        )

    P.W_stmt_reason_state = 0.1 * random_init(sentence_rep_size,2 * sentence_rep_size)
    P.b_reason_state = zeros_init(2 * sentence_rep_size)

    prepare_lookup = build_lookup(
            P,
            input_size=sentence_rep_size,
            key_size=sentence_rep_size * 2
        )
    reason = build_reasoner(
            P,
            input_size=sentence_rep_size,
            hidden_size=sentence_rep_size * 2
        )

    P.W_output_hidden = 0.1 * random_init(sentence_rep_size * 2,sentence_rep_size)
    P.b_output_hidden = zeros_init(sentence_rep_size)
    P.W_output = 0.0 * random_init(sentence_rep_size,output_size)
    P.b_output = zeros_init(output_size)

    def qa(stmt_idxs):
        # stmt_idxs: batch_size x time
        stmt_word_reps = P.vocab[stmt_idxs] # batch_size x time x word_rep_size
        sentence_reps  = encode_stmts(stmt_word_reps)
        story_reps = sentence_reps[:-1]
        question_rep = sentence_reps[-1]
        lookup = prepare_lookup(story_reps)

        reasoning_state = T.tanh(T.dot(question_rep,P.W_stmt_reason_state) + P.b_reason_state)
        attention = [None] * evidence_count
        prev_attention = 0
        for i in xrange(evidence_count):
            attention[i] = lookup(reasoning_state,prev_attention)

            reasoning_state = T.dot(
                        attention[i],
                        reason(reasoning_state,story_reps)
                    )
            prev_attention = attention[i]
        
        output_hidden = T.tanh(T.dot(reasoning_state,P.W_output_hidden) + P.b_output_hidden)
        final_scores = T.dot(output_hidden,P.W_output) + P.b_output
        return T.nnet.softmax(final_scores.dimshuffle('x',0))[0], attention
    return qa

