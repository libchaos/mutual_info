import numpy as np
import pandas as pd
import scipy.sparse as sp

import relatedness as r
r = reload(r)

def approx_equal(a, b, tol=.001):
	return np.abs(a - b) < tol

def test_calc_mutual_info():

	n11 = 49
	n01 = 141
	n10 = 27652
	n00 = 774106

	mi = r.calc_mutual_info(n11, n01, n10, n00)
	soln_mi = .0001105

	assert approx_equal(mi, soln_mi)

	return mi, soln_mi

def test_get_mutual_info_inputs():

	class_freq = np.array([801758, 190])
	term_class_freq = np.zeros((1,2))
	term_class_freq[0,0] = 27652
	term_class_freq[0,1] = 49

	n11, n01, n10, n00 = r.get_mutual_info_inputs(class_freq, term_class_freq)

	assert (n11 == np.array([49])).all()
	assert (n01 == np.array([141])).all()
	assert (n10 == np.array([27652])).all()
	assert (n00 == np.array([774106])).all()

	return n11, n01, n10, n00
