"""
Summarizer: implementations of SumBasic and Latent Semantic Analysis algorithms.
Includes a suite of testing tools, which allows for taking a text or series of
texts and calculating precision, recall, and frequency for both algorithms
using the ROUGE package. Also includes a scraping tool (which relies on Goose)
for downloading articles from online news sources for training and/or testing.

All code was written by the author below except the Python bindings for the 
ROUGE package (originally in Perl). The Python bindings are modified to compute
the ROUGE-L score for a series of test cases.

Author: Brandon Sim

README: [Usage]
To run test cases, place them in the directory PythonROUGE/Tests, with each 
HumanX.txt file containing the gold standard and originalX.txt file containing 
the corresponding original article. Then, run test() in summarize.py to 
generate summarized versions using both SumBasic and LSA algorithms, which 
will be outputted to SumBasicX.txt and LSAX.txt in the same directory 
(PythonROUGE/Tests). Then, run PythonROUGE.py, which is located in the 
PythonROUGE folder, which is a Python wrapper for the Perl ROUGE library, 
and the results will appear in the file PythonROUGE/Tests/ROUGE_results_out.txt
in the following format: 
	filename_tested
	recall precision F-value (alpha = 0.5)

To run summaries of individual articles, use the functions:
	summarize_sumbasic(article, news_corpus, summary_length)
	summarize_lsa(article, news_corpus, summary_length)

To generate a new background corpus or to add documents to the training corpus,
use:
	download_online_corpus('corpora/corpus_urls.txt', OUTFILE)
where corpus_urls.txt is a file containing a URL of an article on each line
and OUTFILE is the output file.

Notes for future: make into a class for better modularization
Further optimize code [current optimizations are a bit 'hackish']
"""
import nltk
import math
import sys
import numpy as np

"""TOKENIZATION TOOLS"""
def tokenize(document):
	"""
	Function which tokenizes a document using Python's NLTK
	(Natural Language ToolKit) word_tokenize function.
	word_tokenize must be fed one sentence at a time, which is the purpose
	of this function.

	Args:
        document: a string that contains the document to be tokenized
    Returns:
        a list of tokens, with punctuation characters removed
    Raises:
        None
	"""
	tokens = []
	sentences = nltk.tokenize.sent_tokenize(document)
	for s in sentences:
		tokens = tokens + nltk.word_tokenize(s)
	return removepunctuation([t.lower() for t in tokens])

def removepunctuation(l):
	"""
	Removes punctuation symbols given a list of tokens.

	Args:
		l: a list of tokens
	Returns:
		a list of tokens without punctuation symbols, as defined in function
	Raises:
		None
	"""
	punctuation = [",", ".", "`", "``", "'", "''", "'s"]
	cleaned = []
	for x in l:
		if x not in punctuation:
			cleaned.append(x)
	return cleaned


"""
CORPUS GENERATION/IMPORTING TOOLS
Use either import_background_corpus OR download_online_corpus, the former for
using the background dataset in an XML file, latter for downloading a 
Goose-derived news article dataset (I curated these URL's).
The rest are helper methods.

Use preload_corpus to prevent having to recalculate frequencies every time
tf_idf is calculated.
"""

def load_corpus(filename):
	"""
	Helper function for preload_corpus. Loads a corpus file.

	Args:
		filename: filename of corpus
	Returns:
		a list of corpus texts
	Raises:
		Exception if file not found
	"""
	corpus = []
	with open(filename) as f:
		for line in f:
			corpus.append(line)
	return corpus

def preload_corpus(c):
	"""
	Preloads frequency distributions of the corpus for efficiency.
	Stores these distributions in cache so we do not have to load
	the corpus every time tf_idf is calculated.
	Relies on the helper function load_corpus.

	Args:
		c: the filename containing the corpus
	Returns:
		a list of nltk FreqDist objects, which contain frequencies of keys
		in corpus
	Raises:
		None
	"""
	preloadedfreq = []
	corpus = load_corpus(c)
	for doc in corpus:
		tokens = tokenize(doc)
		fdist = nltk.FreqDist(tokens)
		preloadedfreq.append(fdist)
	return preloadedfreq

def import_background_corpus(filename):
	"""
	Imports background corpus located in the newsspace200_cleaned dataset.
	(German translations)
	Final implementation does not use this dataset.

	Args:
		filename: filename of corpus
	Returns:
		a list with each element corresponding to one document in corpus
	Raises:
		Exception if file not found
	"""
	descriptions = []
	opentag = '<description>'
	closetag = '</description>'
	with open(filename) as f:
		for li in f:
			descriptions.append(li.split(opentag)[1].split(closetag)[0])	
	return descriptions

def load_article(url):
	"""
	Loads text article from an online link using the goose library.
	Helper function for download_online_corpus.

	Args:
		url: URL of article to load
	Returns:
		a string containing the cleaned text of the article, if retrievable
	Raises:
		Error if article is not parseable
	"""
	print url
	g = Goose()
	article = g.extract(url=url)
	return article.cleaned_text

def download_online_corpus(urlfile, outfile):
	"""
	Loads a set of articles from a set of URL's. Writes output to a text file.
	Run this only once, afterwards load from text file for speed.
	Uses load_article as a helper function.

	Args:
		urlfile: name of text file containing a URL per line for corpus sources
		outfile: name of text file for output
	Returns:
		None
	Raises:
		Error if file is not found or cannot be written to.
	"""
	urls = []
	with open(urlfile) as furl:
		for lineurl in furl:
			urls.append(lineurl)

	with open(outfile, 'w') as f:
		for url in urls:
			f.write(load_article(url).replace('\n', ' '))
			f.write('\n')

""" TERM FREQUENCY-INVERSE DOCUMENT FREQUENCY TOOLS """
def tf(t, d):
	""" 
	Calculates the raw term frequency of a term in a document.

	Args:
		t: term to compute TF of
		d: document that the term is in, tokenized
	Returns: 
		returns a float representing the raw term frequency
	Raises:
		None
	"""
	fdist = nltk.FreqDist(d)
	return float(fdist[t])

def log_tf(t, d):
	"""
	Calculates the log term frequency of a term in a document.
	Adds 1 to prevent taking the log of 0.

	Args:
		t: term to compute log TF of
		d: document that the term is in, tokenized
	Returns:
		returns a float representing the log raw term frequency
	Raises:
		None
	"""
	return math.log(tf(t, d) + 1.0)

def norm_tf(t, d):
	"""
	Calculates normalized term frequency of a term in a document.
	Divides term frequency by maximum term frequency of any term.

	Args:
		t: term to compute normalized term frequency of
		d: document that the term is in, tokenized
	Returns:
		returns a float representing the normalized term frequency
	Raises:
		None
	"""
	fdist = nltk.FreqDist(d)
	v = list(fdist.values())
	return fdist[t]/float(max(v))

def idf(t, c, preloaded, preloadedfreq=0):
	"""
	Calculates the inverse document frequency of a term in a document compared
		to a corpus of documents.
	Calculates log(D / (1+d(t))), where D is the number of documents in corpus
		c, and d(t) is the number of documents that contain term t.
	Allows the user to specify a preloaded list of frequency distributions.
	If preloaded = true, then preloadedfreq must be included, the output from
		preload_corpus.
	If preloaded = false, then we will load the corpus in the method
		(but this is slow!)

	Args:
		t: term to compute inverse document frequency of
		c: corpus location (not needed if preloaded is True)
		preloaded: boolean value determining if preloaded distribution is
			included
		preloadedfreq: optional parameter. should be set if preloaded=True
	Returns:
		returns a float representing the inverse document frequency of term t
	Raises:
		Error if preloaded = True but no preloadedfreq is specified
	"""
	if not preloaded:
		corpus = load_corpus(c)
		numdocuments = len(corpus)
		documents_containing = 0 # documents containing word
		
		for doc in corpus:
			tokens = tokenize(doc)
			fdist = nltk.FreqDist(tokens)
			if fdist[t] > 0:
				documents_containing = documents_containing + 1
		return math.log(numdocuments / (1.0 + documents_containing))
	elif preloaded:
		if preloadedfreq == 0:
			sys.exit(("Error: Preloaded corpus option was selected, "
				"but no frequency distributions were given."))
		else:
			numdocuments = len(preloadedfreq)
			documents_containing = 0

			for dist in preloadedfreq:
				if dist[t] > 0:
					documents_containing = documents_containing + 1
			return math.log(numdocuments / (1.0 + documents_containing))

def tf_idf(t, d, c, preloaded, preloadedfreq=0):
	"""
	Calculates the TF*IDF score of a term in a document
	Allows the user to specify a preloaded list of frequency distributions.
	If preloaded = true, then preloadedfreq must be included, the output from
		preload_corpus.
	If preloaded = false, then we will load the corpus in the method
		(but this is slow!)

	Args:
		t: term to compute inverse document frequency of
		d: document the term is in
		c: corpus location (not needed if preloaded is True)
		preloaded: boolean value determining if preloaded distribution is
			included
		preloadedfreq: optional parameter. should be set if preloaded=True
	Returns:
		returns a float representing the inverse document frequency of term t
	Raises:
		Error if preloaded = True but no preloadedfreq is specified
	"""
	t = t.lower()
	return tf(t, d)*idf(t, c, preloaded, preloadedfreq)

def reorder_sentences(output_sentences, input):
	"""
	Reorders sentences "chronologically" in the order they appear in the
		article
	
	Args:
		output_sentences: a list of sentences to be reordered
		input: the article that the sentences come from
	Returns:
		a list of sentences sorted in chronological order
	Raises:
		None
	"""
	output_sentences.sort(lambda s1,s2: input.find(s1) - input.find(s2))
	return output_sentences

""" SUMBASIC SUMMARIZER IMPLEMENTATION """

def summarize_sumbasic(document, corpus, charcount):
	"""
	Summarizes a text to approximately charcount characters using the SUMBASIC
	algorithm, as described in the accompanying paper and in the literature.
	Basically, SUMBASIC scores each sentence with the average of the TF-IDF
	scores of its words, then greedily finds the highest scoring sentences 
	containing the highest scoring words (key words).

	Args:
		document: filename of document to be summarized
		corpus: filename of corpus to use for IDF weighting
		charcount: an integer containing an approximate of the number of
			characters desired in the summary
	Returns:
		a string containing the generated summary
	Raises:
		Error if files cannot be opened
	"""
	with open(document) as f:
		article = ''
		for line in f:
			article = article + line
		article_tokens = tokenize(article)

		fdist = nltk.FreqDist(article_tokens)
		keys = list(fdist.keys())

		# preload for speed
		preloaded = preload_corpus(corpus)
		tfidf = {}

		for k in keys:
			tfidf[k] = tf_idf(k, article_tokens, corpus, True, preloaded)

		# list of sorted words, most important first
		sortedwords = sorted(tfidf, key=tfidf.get, reverse=True)

		scores = {} # holds the score for each sentence
		sentences = nltk.tokenize.sent_tokenize(article)
		sentences_lower = [se.lower() for se in sentences]

		for s in sentences_lower:
			# score sentence: average of tfidf scores
			words = nltk.word_tokenize(s)
			score = 0
			for w in words:
				score = score + tf_idf(w, article_tokens, corpus, 
					True, preloaded)
			scores[s] = score * 1.0 / len(words)

		# greedy search, picks best scoring sentence for highest scoring word
		# until character count has been reached
		currentchars = 0
		currenttopic = 0
		summary = []
		while currentchars < charcount:
			curmax = 0
			curmaxindex = 0
			for index, candidate in enumerate(sentences_lower):
				if (sortedwords[currenttopic] in nltk.word_tokenize(candidate)
					and scores[candidate] > curmax):
					curmax = scores[candidate]
					curmaxindex = index
			# adds the best scoring sentence to summary
			chosen = sentences_lower[curmaxindex]
			# adds original sentences, not lowercase, to summary
			summary.append(sentences[curmaxindex]) 
			# arbitrary bad score, to prevent sentence chosen twice
			scores[chosen] = -10000

			currenttopic = currenttopic + 1
			currentchars = currentchars + len(chosen)	
		return " ".join(reorder_sentences(summary, article))

""" LATENT SEMANTIC ANALYSIS SUMMARIZER IMPLEMENTATION """
def summarize_lsa(document, corpus, charcount):
	"""
	Summarizes a text to approximately charcount characters using the LSA
	algorithm, as described in the accompanying paper and in the literature.
	Basically, LSA constructs a matrix of terms and sentences, finds the
	singular value decomposition (SVD), and uses the orthogonal basis
	produced to score sentences, the norm of the columns in S*V^T.
	Then, the highest scoring sentences are chosen for the summary.

	Args:
		document: filename of document to be summarized
		corpus: filename of corpus to use for IDF weighting
		charcount: an integer containing an approximate of the number of
			characters desired in the summary
	Returns:
		a string containing the generated summary
	Raises:
		Error if files cannot be opened
	"""
	LSAmatrix = lsa_topic_representation(document, corpus)
	# singular value decomposition
	U, s, Vt = np.linalg.svd(LSAmatrix, full_matrices=False)
	S = np.diag(s)
	D = np.transpose(np.dot(S, Vt))

	with open(document) as f:
		article = ''
		for line in f:
			article = article + line
	sentences = nltk.tokenize.sent_tokenize(article)

	weights = {}

	for rownum, row in enumerate(D):
		weights[sentences[rownum]] = float(np.linalg.norm(row)) # 2-norm

	sortedsentences = sorted(weights, key=weights.get, reverse=True)
	#sortedweights = sorted(weights.values(), reverse=True)

	currentchars = 0
	curindex = 0
	summary = []
	# chooses highest scoring sentences until character count met
	while currentchars < charcount:
		chosen = sortedsentences[curindex]
		summary.append(chosen)
		currentchars = currentchars + len(chosen)
		curindex = curindex + 1

	return " ".join(reorder_sentences(summary, article))

def lsa_topic_representation(document, corpus):
	"""
	Constructs the term/sentences matrix for use in the LSA algorithm.
	Each entry in the matrix is 0 if the term is not present in the 
	corresponding sentence, or the TF-IDF weight of the term if it is
	present in the sentence.

	Args:
		document: filename of document to be summarized
		corpus: filename of corpus to use for IDF weighting
	Returns:
		a matrix of terms and sentences for use in the LSA algorithm
	Raises:
		Error if files cannot be opened
	"""
	with open(document) as f:
		article = ''
		for line in f:
			article = article + line
		article_tokens = tokenize(article)

		fdist = nltk.FreqDist(article_tokens)
		keys = list(fdist.keys())

		# preload for speed
		preloaded = preload_corpus(corpus)
		
		sentences = nltk.tokenize.sent_tokenize(article)
		sentences_lower = [se.lower() for se in sentences]

		numwords = len(keys)
		numsentences = len(sentences)

		LSAmatrix = np.zeros((numwords, numsentences))

		for wordi, w in enumerate(keys):
			for senti, s in enumerate(sentences_lower):
				# if word is in sentence weight is TF*IDF, else 0
				if w in nltk.word_tokenize(s):
					LSAmatrix[wordi][senti] = tf_idf(w, article_tokens, 
						corpus, True, preloaded)
				else:
					LSAmatrix[wordi][senti] = 0
		return LSAmatrix

""" 
TESTING MODULE 
Run this function to generate the summaries algorithmically before
evaluating them with PythonROUGE.py.
"""

def test_ROUGE():
	"""
	Provides a testing module to calculate ROUGE scores for a summary.
	Given a set of original documents, produces summaries using both SUMBASIC
	and LSA algorithms and writes out to file for preparation before using
	PythonROUGE.py to analyze.

	Args:
		None
	Returns:
		None
	Raises:
		Errors if files cannot be opened
	"""
	news_corpus = 'corpora/news_corpus.txt'
	originalbase = 'PythonROUGE/Tests/original'
	tests = 'PythonROUGE/Tests/'
	offset = 0 # number of extra words (control precision/recall tradeoff)
	numtests = 20 # hardcoded for now
	for i in range(numtests):
		original = originalbase + str(i+1) + '.txt'

		with open(tests + 'Human' + str(i+1) + '.txt') as fhs:
			humansum = ''
			for hs in fhs:
				humansum = humansum + hs
			len_human = len(humansum)

		with open(tests + 'SumBasic' + str(i+1) + '.txt', 'w') as fout:
			sumbasic = summarize_sumbasic(original, news_corpus, 
				len_human+offset)
			print sumbasic
			fout.write(sumbasic)
		with open(tests + 'LSA' + str(i+1) + '.txt', 'w') as fout2:
			lsa = summarize_lsa(original, news_corpus, len_human+offset)
			print lsa
			fout2.write(lsa)

def main():
	"""
	Summarizes the article given in the variable article using both algorithms.
	Useful for single uses and as example cases for usage of the summarization
	methods.

	Args:
		None
	Returns:
		None
	Raises:
		Errors if files cannot be opened.
	"""
	# the following command only needs to be run once to gather corpus
	#download_online_corpus('corpora/corpus_urls.txt', 'news_corpus.txt')
	news_corpus = 'corpora/news_corpus.txt'
	article = 'articles/article2.txt'
	#article = 'inaugural/2009-Obama.txt'

	#gets length of article in characters
	with open(article) as f:
		a = ''
		for line in f:
			a = a + line

	summary_length = 500
	
	# print results
	print "Length of article in characters: " + str(len(a))
	print "SumBasic summarizer: " + str(summary_length) + " characters"
	print summarize_sumbasic(article, news_corpus, summary_length)
	print "\n"
	print "LSA summarizer: " + str(summary_length) + " characters"
	print summarize_lsa(article, news_corpus, summary_length)

if __name__ == "__main__":
	test_ROUGE()
	#main()
