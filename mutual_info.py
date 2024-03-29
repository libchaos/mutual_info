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

	assert n11 >= 0 
	assert n01 >= 0
	assert n10 >= 0
	assert n00 >= 0

	# add a small amount of mass to zeroes so that MI is not undefined
	if n11 == 0:
		n11 = .5
	if n01 == 0:
		n01 = .5
	if n10 == 0:
		n10 = .5
	if n00 == 0:
		n00 = .5

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
		here they are arrays. for instance n11[i,j] = # of documents in the jth class where the ith term appears  
	'''
	
	n_terms, n_classes = term_class_freq.shape

	n11 = np.zeros((n_terms, n_classes))
	n01 = np.zeros((n_terms, n_classes))
	for i in range(n_terms):
		for j in range(n_classes):
			n11[i,j] = term_class_freq[i,j]
			n01[i,j] = class_freq[j] - n11[i,j]
 
	n10 = np.zeros((n_terms, n_classes))
	n00 = np.zeros((n_terms, n_classes))
	for i in range(n_terms):
		for j in range(n_classes):
			n10[i,j] = np.sum(n11[i,:]) - n11[i,j]
			n00[i,j] = np.sum(n01[i,:]) - n01[i,j] 	

	return n11, n01, n10, n00

def get_term_class_index(class_freq_df, term_class_freq_df):
	'''Create a dictionary mapping term and class ids to an index
	to be used in creating arrays (see get_freq_from_df)
	'''

	termids = np.unique(term_class_freq_df.termid.values)
	classids = np.unique(term_class_freq_df.classid.values)

	termid_index = {}
	for i, termid in enumerate(termids):
		termid_index[termid] = i

	classid_index = {}
	for i, classid in enumerate(classids):
		classid_index[classid] = i

	return termid_index, classid_index

def get_freq_from_df(class_freq_df, term_class_freq_df, termid_index, classid_index):
	'''Convert data frames of class frequencies and class-term frequencies 
	to arrays of those values

	Args:
		class_freq_df (pd.DataFrame):
			see README (example_class_freq.txt)
		term_class_freq_df (pd.DataFrame):
			see README (example_term_class_freq.txt)
		termid_index (dict):
			each key is a termid and each value gives the index associated with that termid in the returned arrays
		classid_index (dict):
			each key is a classid and each value gives the index associated with that classid in the returned arrays

	Return:
		class_freq (ndarray): see get_mutual_info_inputs
		term_class_freq (ndarray): see get_mutual_info_inputs
	'''
	
	class_freq = np.zeros(len(classid_index))
	for i in range(len(class_freq_df)):
		classid = class_freq_df['classid'][i]
		n = class_freq_df['n'][i]
		class_freq[classid_index[classid]] = n

	term_class_freq = np.zeros((len(termid_index), len(classid_index)))
	for i in range(len(term_class_freq_df)):
		termid = term_class_freq_df['termid'][i]
		classid = term_class_freq_df['classid'][i]
		n = term_class_freq_df['n'][i]

		term_class_freq[termid_index[termid], classid_index[classid]] = n

	return class_freq, term_class_freq

def calc_mutual_info_df(class_freq_df, term_class_freq_df, termid_index, classid_index):
	'''Calculate mutual info from data frames of class and term-class frequencies
	'''

	class_freq, term_class_freq = get_freq_from_df(class_freq_df, term_class_freq_df, termid_index, classid_index)
	n11, n01, n10, n00 = get_mutual_info_inputs(class_freq, term_class_freq)

	n_terms, n_classes = n11.shape

	mi = np.zeros((n_terms, n_classes))
	for i in range(n_terms):
		for j in range(n_classes):
			mi[i,j] = calc_mutual_info(n11[i,j], n01[i,j], n10[i,j], n00[i,j])

	return mi

def mutual_info(class_freq_filename, term_class_freq_filename):

	class_freq_df = pd.read_csv(class_freq_filename, sep='\t')	
	term_class_freq_df = pd.read_csv(term_class_freq_filename, sep='\t')

	termid_index, classid_index = get_term_class_index(class_freq_df, term_class_freq_df)
	mi = calc_mutual_info_df(class_freq_df, term_class_freq_df, termid_index, classid_index)

	mis = []
	termids = []
	classids = []	
	for termid in termid_index.keys():
		for classid in classid_index.keys():
			termids.append(termid)
			classids.append(classid)
			mis.append(mi[termid_index[termid], classid_index[classid]])	
	
	mi_df = pd.DataFrame({'termid': termids, 'classid': classids, 'mi': mis}).sort(['classid','mi'], ascending=[1,0])

	return mi_df

if __name__ == '__main__':

	# define command line usage
	usage = "usage: %prog [options] class_freq_filename term_class_freq_filename mutual_info_filename"
	parser = optparse.OptionParser(usage)
	(options, args) = parser.parse_args()

	assert len(args) == 3
	
	class_freq_filename = args[0]
	term_class_freq_filename = args[1]
	mutual_info_filename = args[2]

	mi_df = mutual_info(class_freq_filename, term_class_freq_filename)

	mi_df.to_csv(mutual_info_filename, index=False, sep='\t')
			
