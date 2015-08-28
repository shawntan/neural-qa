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
    P.W_lookup_input_hidden = random_init(input_size,hidden_size)
    P.W_lookup_key_hidden = random_init(key_size,hidden_size)
    P.b_lookup_hidden = zeros_init(hidden_size)
    P.W_lookup = zeros_init(hidden_size)
    def prepare_lookup(sentence_reps):
        _hidden = T.dot(sentence_reps,P.W_lookup_input_hidden)
        def lookup(key):
            score = T.dot(T.tanh(
                    T.dot(key,P.W_lookup_key_hidden) +\
                    _hidden + P.b_lookup_hidden
                ),P.W_lookup)
            return T.nnet.softmax(score.dimshuffle('x',0))[0]
        return lookup
    return prepare_lookup

def build_reasoner(P,input_size,hidden_size):
    P.W_reasoner_input_hidden = random_init(input_size,hidden_size)
    P.W_reasoner_hidden_hidden = random_init(hidden_size,hidden_size)
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

    prepare_lookup = build_lookup(
            P,
            input_size=sentence_rep_size,
            key_size=sentence_rep_size
        )
    reason = build_reasoner(
            P,
            input_size=sentence_rep_size,
            hidden_size=sentence_rep_size
        )
    P.W_output = random_init(sentence_rep_size,output_size)
    P.b_output = zeros_init(output_size)

    def qa(stmt_idxs):
        # stmt_idxs: batch_size x time
        stmt_word_reps = P.vocab[stmt_idxs] # batch_size x time x word_rep_size
        sentence_reps  = encode_stmts(stmt_word_reps)
        story_reps = sentence_reps[:-1]
        question_rep = sentence_reps[-1]
        lookup = prepare_lookup(story_reps)

        reasoning_state = question_rep
        attention = [None] * evidence_count
        for i in xrange(evidence_count):
            attention[i] = lookup(reasoning_state)
            evidence = T.dot(attention[i],story_reps)
            reasoning_state = reason(reasoning_state,evidence)
        
        final_scores = T.dot(reasoning_state,P.W_output) + P.b_output
        return T.nnet.softmax(final_scores.dimshuffle('x',0))[0]
    return qa

if __name__ == "__main__":
    P = Parameters()
    vocab_size = 53
    qa = build(P,
            vocab_size=vocab_size,
            word_rep_size=vocab_size,
            sentence_rep_size=128,
            output_size=vocab_size,
            evidence_count=2
        )
    story = np.array([[-1, 25, 34, 45, 43,  7,  0],
                      [-1, 41, 26, 45, 43, 37,  0],
                      [-1, 41, 34, 45, 43, 28,  0],
                      [-1, 10, 47, 45, 43,  7,  0],
                      [25, 50,  5, 45, 43, 37,  0],
                      [-1, 32, 47, 45, 43, 18,  0],
                      [-1, 41, 34, 45, 43, 22,  0],
                      [-1, 41, 50, 45, 43, 18,  0],
                      [25, 50,  5, 45, 43, 28,  0],
                      [-1, 10, 50, 45, 43, 37,  0],
                      [-1, 41, 34, 45, 43, 22,  0],
                      [10, 38, 48, 43,  3, 44,  0],
                      [-1, 32, 50, 45, 43,  7,  0],
                      [10, 50,  5, 45, 43, 18,  0],
                      [-1, -1, 10, 30, 43,  3,  0],
                      [-1, 10, 20, 43,  3, 44,  0],
                      [-1, 41, 46, 43, 16, 44,  0],
                      [-1, -1, 10, 11, 43,  3,  0],
                      [-1, 32, 26, 45, 43, 28,  0],
                      [-1, 41, 26, 45, 43, 28,  0],
                      [-1, -1, 51, 24, 43,  3,  1]],dtype=np.int32)
    print qa(story).eval()




