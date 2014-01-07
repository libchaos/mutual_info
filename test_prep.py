import prep
prep = reload(prep)

def test_prep():

  doc_stream = ['1\ta\thello there\n', '2\tb\tyo yo\n', '3\ta\thi hello\n', '4\ta\tyo\n']
  class_freq_text, term_class_freq_text = prep.prep(doc_stream)
 
  soln_class_freq_text = 'classid\tn\na\t3\nb\t1\n'
  soln_term_class_freq_text = 'termid\tclassid\tn\nhi\ta\t1\nthere\ta\t1\nhello\ta\t2\nyo\ta\t1\nyo\tb\t1\n' 
  assert class_freq_text == soln_class_freq_text
  assert term_class_freq_text == soln_term_class_freq_text

