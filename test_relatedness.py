import numpy as np

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
