import numpy as np
import pandas as pd

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

	assert (n11 == np.array([[27652, 49]])).all()
	assert (n01 == np.array([[774106, 141]])).all()
	assert (n10 == np.array([[49, 27652]])).all()
	assert (n00 == np.array([[141, 774106]])).all()

	return n11, n01, n10, n00

def test_get_freq_from_df():

	class_freq_df = pd.DataFrame({'classid': [0, 1], 'n': [801758, 190]})
	term_class_freq_df = pd.DataFrame({'termid': ['export','export'], 'classid': [0, 1], 'n': [27652, 49]})

	termid_index, classid_index = r.get_term_class_index(class_freq_df, term_class_freq_df)
	class_freq, term_class_freq = r.get_freq_from_df(class_freq_df, term_class_freq_df, termid_index, classid_index)

	soln_class_freq = np.array([801758, 190])
	soln_term_class_freq = np.zeros((1,2))
	soln_term_class_freq[0,0] = 27652
	soln_term_class_freq[0,1] = 49

	assert (class_freq == soln_class_freq).all()
	assert (term_class_freq == soln_term_class_freq).all()

	return vars()

def test_calc_mutual_info_df():

	class_freq_df = pd.DataFrame({'classid': [0, 1], 'n': [801758, 190]})
	term_class_freq_df = pd.DataFrame({'termid': ['export','export'], 'classid': [0, 1], 'n': [27652, 49]})

	termid_index, classid_index = r.get_term_class_index(class_freq_df, term_class_freq_df)
	mi = r.calc_mutual_info_df(class_freq_df, term_class_freq_df, termid_index, classid_index)
	soln_mi = np.array([[.0001105, .0001105]])

	assert mi.shape == soln_mi.shape
	assert approx_equal(mi[0,0], soln_mi[0,0])
	assert approx_equal(mi[0,1], soln_mi[0,1])

	return mi, soln_mi	
