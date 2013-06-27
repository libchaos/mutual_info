import numpy as np
import pandas as pd
import optparse
import pdb

def calc_mutual_info(n11, n01, n10, n00):
	'''Calculate the mutual information
	(http://nlp.stanford.edu/IR-book/html/htmledition/mutual-information-1.html)

	Args:
		n11 (int): # of documents in the class where the term appears
		n10 (int): # of documents not in the class where the term appears
		n01 (int): # of documents in the class where the term does not appear
		n00 (int): # of documents not in the class where the term does not appear 
	'''

	n = float(n11 + n10 + n01 + n00)
	n1x = n11 + n10
	nx1 = n11 + n01
	n0x = n01 + n00
	nx0 = n00 + n10

	mi = 0
	mi += (n11/n)*np.log2((n*n11)/(n1x*nx1))
	mi += (n01/n)*np.log2((n*n01)/(n0x*nx1))
	mi += (n10/n)*np.log2((n*n10)/(n1x*nx0))
	mi += (n00/n)*np.log2((n*n00)/(n0x*nx0))

	return mi

def get_mutual_info_inputs(class_freq, term_class_freq):
	'''Convert arrays of class frequencies and class-term frequencies to counts 
	used in the calculation of mutual info

	Args: 
		class_freq (ndarray):
			class_freq[i] = # of documents in the ith class
		term_class_freq (ndarray):
			term_class_freq[i,j] = # of documents in the ith class where the jth term appears
	
	Return:
		see calc_mutual_info except that inputs to calc_mutual_info are integers and 
		here they are arrays 
	'''

	n11 = term_class_freq[:,1]
	n01 = class_freq[1] - n11
	n10 = term_class_freq[:,0]
	n00 = class_freq[0] - n10
	
	return n11, n01, n10, n00

def get_freq_from_df(class_freq_df, term_class_freq_df):
	'''Convert data frames of class frequencies and class-term frequencies 
	to arrays of those values

	Args:
		class_freq_df (pd.DataFrame):
			see README (example_class_freq.txt)
		term_class_freq_df (pd.DataFrame):
			see README (example_term_class_freq.txt)

	Return:
		class_freq (ndarray): see get_mutual_info_inputs
		term_class_freq (ndarray): see get_mutual_info_inputs
	'''
	
	class_freq = class_freq_df.n.values

	termids = np.unique(term_class_freq_df.termid.values)

	termid_index = {}
	for i, termid in enumerate(termids):
		termid_index[termid] = i

	term_class_freq = np.zeros((len(termids), 2))
	for i in range(len(term_class_freq_df)):
		termid = term_class_freq_df['termid'][i]
		classid = term_class_freq_df['classid'][i]
		n = term_class_freq_df['n'][i]

		term_class_freq[termid_index[termid], classid] = n

	return class_freq, term_class_freq

def calc_mutual_info_df(class_freq_df, term_class_freq_df):
	'''Calculate mutual info from data frames of class and term-class frequencies
	'''

	class_freq, term_class_freq = get_freq_from_df(class_freq_df, term_class_freq_df)
	n11, n01, n10, n00 = get_mutual_info_inputs(class_freq, term_class_freq)

	mi = np.zeros(len(n11))
	for i in range(len(n11)):
		mi[i] = calc_mutual_info(n11[i], n01[i], n10[i], n00[i])

	return mi

def calc_relatedness(class_freq_filename, term_class_freq_filename):

	class_freq_df = pd.read_csv(class_freq_filename, sep='\t')	
	term_class_freq_df = pd.read_csv(term_class_freq_filename, sep='\t')

	mi = calc_mutual_info_df(class_freq_df, term_class_freq_df).tolist()
	classids = class_freq_df.classid.values.tolist()

	relatedness = pd.DataFrame({'classid': classids[1], 'r': mi}).sort('r', ascending=0)

	return relatedness

if __name__ == '__main__':

	# define command line usage
	usage = "usage: %prog [options] class_freq_filename term_class_freq_filename relatedness_filename"
	parser = optparse.OptionParser()
	(options, args) = parser.parse_args()

	assert len(args) == 3
	
	class_freq_filename = args[0]
	term_class_freq_filename = args[1]
	relatedness_filename = args[2]

	relatedness = calc_relatedness(class_freq_filename, term_class_freq_filename)

	relatedness.to_csv(relatedness_filename, index=False, sep='\t')
			
