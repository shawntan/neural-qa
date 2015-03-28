import data_io
import sys
import cPickle as pickle
from collections import Counter
question_words = ["who","what","when","where","why","how"]
def load(filename):
	vocab = pickle.load(open(filename,'rb'))
	vocab = { i:k for k,i in enumerate(vocab) }
	return vocab

if __name__ == "__main__":
	filenames = sys.argv[1:-1]
	output_file = sys.argv[-1]
	input_vocab = set(question_words)
	for filename in filenames:
		entities = set(r[0] for _,r in data_io.process_lines(filename,lower=False)
								if r != None)
		for input_sentence,_ in data_io.process_lines(filename,lower=False):
			entities.update(
					i.lower() for pos,i in enumerate(input_sentence[1])
						if i[0].isupper() and i.lower() not in input_vocab 
					)
		print entities
		for input_sentence,_ in data_io.process_lines(filename,lower=True):
			input_vocab.update(i for i in input_sentence[1] if i not in entities)
	print sorted(input_vocab)
	pickle.dump(sorted(input_vocab),open(output_file,'wb'),2)
