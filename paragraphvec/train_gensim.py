# import libraries
import gensim
import random
import pandas as pd
import numpy as np
from itertools import combinations

# import implemented python files
import models_gensim

class Ingredient2Vec:
	def __init__(self):
		print("\n\n...Ingredient2Vec initialized")

	def build_taggedDocument_df(self, df, filtering=5, random_sampling=False, num_sampling=0):

		print('\n\n...Building Training Set')
		for index, row in df.iterrows():
			compound_list = row['text'].split("  ")
			#print compound_list
			ingredient_name = row['label'].decode("utf8").strip()

			# Random Sampling
			if random_sampling:
				#sample randomly
				for i in xrange(num_sampling):
					sampled_compounds = random.sample(compound_list, filtering)
					yield gensim.models.doc2vec.TaggedDocument(compound_list, [ingredient_name])

			# Data as they are
			else:
				yield gensim.models.doc2vec.TaggedDocument(compound_list, [ingredient_name])

if __name__ == '__main__':
	gensimLoader = models_gensim.GensimModels()
	ingr2vec = Ingredient2Vec()

	"""
	Mode Description

	# mode 1 : Embed Ingredients with Chemical Compounds
	# mode 999 : Plot Loaded Word2Vec or Doc2vec
	"""

	mode = 1
	if mode == 1:
		"""
		Load Data
		"""
		#ingredient_sentence = "../data/scientific_report/D7_flavornet-vocab-compounds.csv"
		ingredient_sentence = "../data/scientific_report/flavordb_ver2.0.csv"
		df = pd.read_csv(ingredient_sentence)
		"""
		Preprocess Data

		"""
		# build taggedDocument form of corpus
		corpus_ingr2vec = list(ingr2vec.build_taggedDocument_df(df, filtering=Config.FILTERING, random_sampling=Config.RANDOM_SAMPLING, num_sampling=Config.NUM_SAMPLING))

		"""
		Build & Save Doc2Vec

		"""
		# build ingredient embeddings with doc2vec
		model_ingr2vec = gensimLoader.build_doc2vec(corpus_ingr2vec, load_pretrained=False, path_pretrained=False)

		# save character-level compounds embeddings with doc2vec
		gensimLoader.save_doc2vec_only_doc(model=model_ingr2vec, path=Config.path_embeddings_compounds_rnd)

		model_loaded = gensimLoader.load_word2vec(path=Config.path_embeddings_compounds_rnd)

		#for x in model_loaded.vocab:
		#	print x, model_loaded.word_vec(x)

	elif mode == 999:

		"""
		Plot Ingredient2Vec

		"""
		model_loaded = gensimLoader.load_word2vec(path=Config.path_embeddings_compounds_rnd)
		model_tsne = DataPlotter.load_TSNE(model_loaded, dim=2)
		DataPlotter.plot_category(model_loaded, model_tsne, Config.path_embeddings_compounds_rnd, withLegends=False)
		#DataPlotter.plot_clustering(model_loaded, model_tsne, Config.path_plottings_ingredients_clustering)

	else:
		print("Please specify the mode you want.")
