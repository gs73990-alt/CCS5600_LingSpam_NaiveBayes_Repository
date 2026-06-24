Task 2 Naive Bayes results using Tom M. Mitchell text algorithm

This implementation trains a Naive Bayes text classifier from scratch.
It uses word occurrence counts, class priors, and Laplace smoothing.

Mitchell-style training used here:
P(v_j) = number of documents in class v_j / total training documents
P(w_k | v_j) = (count of word w_k in class v_j + 1) / (total word positions in class v_j + vocabulary size)

Classification uses log probabilities to avoid numerical underflow.
Only words that appear in the training vocabulary are used during classification.

Processed corpus versions:
bare: No lemmatization, stop-words included
stop: No lemmatization, stop-words removed
lemm: Lemmatized, stop-words included
lemm_stop: Lemmatized, stop-words removed

Vocabulary sizes used:
N = 50
N = 100
N = 500
N = 1000
N = 5000
N = all

Main files:
overall_naive_bayes_mitchell_results.csv: best file to inspect first.
combined_naive_bayes_mitchell_fold_results.csv: fold-by-fold results for every version and N value.
Each version folder contains prediction files and vocabulary files.
