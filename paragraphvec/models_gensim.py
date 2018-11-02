import time
import gensim
from gensim.models.keyedvectors import KeyedVectors
import multiprocessing
import pickle

class GensimModels():

	"""
	Train Word2Vec Model with Gensim
	"""

	def build_word2vec(self, corpus, load_pretrained=False, path_pretrained=""):
		print("\n\n...Start to build Word2Vec Models with Gensim")

		time_start = time.time()
		cores = multiprocessing.cpu_count()

		model = gensim.models.Word2Vec(corpus, size=50, window=5, min_count=5, workers=cores)

		print("Word Embedding Dimension:", 50)
		print("Word Window & Filtering:", 5)

		print("Unique Words Count:", len(model.wv.vocab))

		return model

	def save_word2vec(self, model, path):
		print("\n\n...Save Word2Vec with a file name of", path)
		model.wv.save_word2vec_format(path, binary=True)

	def load_word2vec(self, path):
		print("\n\n...Load Word2Vec with a file name of", path)
		model = KeyedVectors.load_word2vec_format(path, binary=True)
		return model

	"""
	Train Doc2Vec Model with Gensim

	"""
	def build_doc2vec(self, corpus, load_pretrained, path_pretrained,
					  # For training
					  doc_dim,
					  word_dim,
					  dm,
					  dm_mean,
					  dm_concat,
					  dbow_words,
					  context_size,
					  lr,
					  epochs,
					  negative_sampling):
		print("\n\n...Start to build Doc2Vec Models with Gensim")

		time_start = time.time()
		cores = multiprocessing.cpu_count()
		model = gensim.models.doc2vec.Doc2Vec(
											  # For training
											  dm=dm, dm_mean=dm_mean, dm_concat=dm_concat, dbow_words=dbow_words,
											  vector_size=doc_dim, alpha=lr, window=context_size, negative_sampling=negative_sampling, epochs=epochs)

		model.build_vocab(corpus)

		if load_pretrained:
			#model_loaded = self.load_word2vec(path_pretrained)
			#print(model_loaded)
			print("...Update Input Vectors with Pre-Trained Vectors:", path_pretrained)
			#model.intersect_word2vec_format(path_pretrained, lockf=0.0, binary=True, encoding='utf8', unicode_errors='strict')

		print("Unique Words Count:", len(model.wv.vocab))
		print("Total Documents Count:", model.corpus_count)


		print("\n\n...Training Started")
		model.train(corpus, total_examples=model.corpus_count, epochs=model.iter)

		print("Doc2Vec training done!")
		print("Time elapsed: {} seconds".format(time.time()-time_start))

		return model

	def save_doc2vec(self, model, path):
		model.save_word2vec_format(path, doctag_vec=True, word_vec=True, prefix='*dt_', fvocab=None, binary=True)

	def save_doc2vec_only_doc(self, model, path):
		model.save_word2vec_format(path, doctag_vec=True, word_vec=False, prefix='', fvocab=None, binary=True)
