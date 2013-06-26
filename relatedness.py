import numpy as np

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
		
