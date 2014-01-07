import optparse
import numpy as np

def prep(doc_stream):

	class_freq = {}
	term_class_freq = {}

	for line in doc_stream:

		vals = line.split('\t')
		
		doc_id = vals[0]
		class_id = vals[1]
		text = vals[2].replace('\n', '').replace('\r', '')

		class_freq[class_id] = class_freq.get(class_id, 0) + 1
		
		if term_class_freq.has_key(class_id) == False:
			term_class_freq[class_id] = {}
	
		words = np.unique(text.split(' '))
		for word in words:
			term_class_freq[class_id][word] = term_class_freq[class_id].get(word, 0) + 1

	# class freq
	class_freq_text = 'classid\tn\n'
	for class_id in class_freq.keys():
		class_freq_text += class_id + '\t' + str(class_freq[class_id]) + '\n'

	# term class freq
	term_class_freq_text = 'termid\tclassid\tn\n'
	for class_id in term_class_freq.keys():
		for term_id in term_class_freq[class_id].keys():
			term_class_freq_text += term_id + '\t' + class_id + '\t' + str(term_class_freq[class_id][term_id]) + '\n'

	return class_freq_text, term_class_freq_text

if __name__ == '__main__':
	
	# define command line usage
	usage = "usage: %prog [options] doc_filename class_freq_filename term_class_freq_filename"
	parser = optparse.OptionParser()
	(options, args) = parser.parse_args()

	doc_filename = args[0]
	class_freq_filename = args[1]
	term_class_freq_filename = args[2]

	f = open(doc_filename, 'r')
	class_freq_text, term_class_freq_text = prep(f)	
	f.close()

	f = open(class_freq_filename, 'w')
	f.write(class_freq_text)
	f.close()

	f = open(term_class_freq_filename, 'w')
	f.write(term_class_freq_text)
	f.close()
