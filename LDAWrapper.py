'''
LDA Wrapper on Gensim's LDA that allows it to be more easily trained by strings, etc

Ankit Kumar
ankit@205consulting.com
'''

import gensim
import nltk
import string



'''
USAGE: inherit this module and overwrite preprocess. Or, copy-paste this module and manually overwrite preprocess. 

you can use it like gensim's lda easily, i.e:

gw = GensimWrapper_205(corpus=['this is a sample corpus','here is more'], num_topics=10)
gw.add_documents(['another corpus','should be an iterable'])

the relevant apis are the __init__(), add_documents(), and query()

query() takes a string and returns a vector representation (in numpy)

this is not yet complete but has most of what you need for typical lda use (training and querying)
'''

class GensimWrapper_205(gensim.models.ldamodel.LdaModel):

	def __init__(self, corpus=None, num_topics=100, id2word=None, distributed=False, chunksize=2000, passes=1, update_every=1, alpha='symmetric', eta=None, decay=0.5, eval_every=10, iterations=50, gamma_threshold=0.001):
		# if corpus is given, it's assumed to be an iterable of strings. so we turn it into a gensim bag. 
		if corpus is not None:
			corpus, id2word = self.generate_gensim_corpus_from_strings(corpus, id2word)
		gensim.models.ldamodel.LdaModel.__init__(self, corpus=corpus, num_topics=num_topics, id2word=id2word, distributed=distributed, chunksize=chunksize, passes=passes, update_every=update_every, alpha=alpha, eta=eta, decay=decay, eval_every=eval_every, iterations=iterations, gamma_threshold=gamma_threshold)
	

	def preprocess(self, doc):
		'''
		baseline preprocessing function; just lowers and splits

				this function should be overwritten, but must always return a list of strings that represent the bag of words of the doc.
		'''
		# remove punctuations
		doc = doc.translate(string.maketrans("",""), string.punctuation)
		# lower and split into a bag of words
		return doc.lower().split()
	def generate_gensim_corpus_from_strings(self, corpus, id2word):

		'''
		this function takes an iterable of strings and turns it into a gensim corpus. if id2word is none, it also creates a gensim dictionary.

		returns the gensim corpus and dictionary (just returns the original dictionary if one is given)
		'''
		# if id2word is none, create a dictionary
		if id2word is None:
			gensim_dict = gensim.corpora.dictionary.Dictionary()
			#update with documents
			gensim_dict.add_documents([self.preprocess(doc) for doc in corpus])
			#re-write over id2word
			id2word = gensim_dict



		

		# now we create the gensim corpus
		gensim_corpus = [id2word.doc2bow(self.preprocess(doc)) for doc in corpus]

		# and return both
		return gensim_corpus, id2word

	def add_documents(self, corpus):
		'''
		params:
			- corpus: iterable of strings, each string considered to be a document

		returns:
			- none; trains the model

		notes:
		this is not called update simply so that it doesn't overwrite gensim's own update function
		'''
		# turn the corpus into a gensim corpus; this will also return an id2word dictionary if one isn't stored in the model yet
		corpus, self.id2word = self.generate_gensim_corpus_from_strings(corpus, self.id2word)
		# train lda model
		self.update(corpus)
		return

	def query(self, query):
		'''
		params:
			- query: a string to query; or a document, for example
		returns:
			- num_topics dimensional vector representing the per-document topic distribution (theta)
		'''
		# preprocess and turn the query into a gensim bag
		gensim_bag = self.id2word.doc2bow(self.preprocess(query))
		# query the lda model
		gamma, sstats = self.lda_model.inference(gensim_bow)
		# normalize the gamma (gamma here is theta)
		normalized_gamma = gamma[0] / gamma[0].sum()
		return normalized_gamma

	def update_dictionary(self, chunk):
		raise NotImplementedError
		''' to do: iterable online making of dictionary + training lda '''

	def update_model(self, chunk):
		raise NotImplementedError
		''' to do ''' 

	'''to do: probability of generation stuff '''


if __name__ == "__main__":
	gw = GensimWrapper_205(['this is a sample corpus','just to see if it works','just three strings'], num_topics=10)
	gw.add_documents(['a sample corpus works','strings if it','break on new string?'])

