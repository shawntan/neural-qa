import data_io
import sys
import cPickle as pickle
from collections import Counter
def load(filename):
	vocab = pickle.load(open(filename,'rb'))
	vocab = { i:k for k,i in enumerate(vocab) }
	return vocab

if __name__ == "__main__":
	filenames = sys.argv[1:-1]
	output_file = sys.argv[-1]
	vocab = set()
	for filename in filenames:
		for i,r in data_io.process_lines(filename):
			vocab.update(i[1])
			if r != None:
				vocab.update(r[0])
	pickle.dump(sorted(vocab),open(output_file,'wb'),2)
