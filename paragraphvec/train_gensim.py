# import libraries
import gensim
import random
import pandas as pd
import numpy as np
import fire
import os
from itertools import combinations

# import implemented python files
import models_gensim, plot

class Preprocessor():
	def build_taggedDocument_df(self, df, random_sampling=False, num_sampling=10, context_size=5):
		print('\n\n...Building Training Set')
		for index, row in df.iterrows():
			compound_list = row['text'].split("  ")
			#print compound_list
			ingredient_name = row['label']

			# Random Sampling
			if random_sampling:
				#sample randomly
				for i in range(num_sampling):
					sampled_compounds = random.sample(compound_list, context_size)
					yield gensim.models.doc2vec.TaggedDocument(compound_list, [ingredient_name])

			# Data as they are
			else:
				yield gensim.models.doc2vec.TaggedDocument(compound_list, [ingredient_name])

	def contract_category(self, ingr2cate):
		rootdir = '../data/'
		file = 'category_mapping'
		file_path = os.path.join(rootdir, file)

		dict_categories = {}
		with open(file_path) as f:
			for line in f.readlines():
				cate_origin = line.split("\t")[0].rstrip()
				cate_target = line.split("\t")[1].rstrip()
				dict_categories[cate_origin] = cate_target

		categories = []
		for ingr in ingr2cate:
			cate_origin = ingr2cate[ingr]

			if cate_origin is 'None':
				ingr2cate[ingr] = 'None'
			else:
				ingr2cate[ingr] = dict_categories[cate_origin]

		for ingr in ingr2cate:
			categories.append(ingr2cate[ingr])
		categories = list(set(categories))
		#print(len(categories))
		#print(sorted(categories))

		return ingr2cate

def start(path_file='flavordb_ver2.0.csv',
		  mode='train',
		  model_ver=0,

		  #datset
		  random_sampling=False,
		  num_sampling=10,
		  load_pretrained=False,
		  path_pretrained="../data/flavordb/id2compound.pkl",

		  #model
		  doc_dim=100,
		  word_dim=50,
		  dm=0,					# If dm=1, ‘distributed memory’ (PV-DM) is used. Otherwise, distributed bag of words (PV-DBOW) is employed.
		  dm_mean=0,			# If 0 , use the sum of the context word vectors. If 1, use the mean. Only applies when dm is used in non-concatenative mode.
		  dm_concat=1,			# If 1, use concatenation of context vectors rather than sum/average; Note concatenation results in a much-larger model, as the input is no longer the size of one (sampled or arithmetically combined) word vector, but the size of the tag(s) and all words in the context strung together.
		  dbow_words=0,			# If set to 1 trains word-vectors (in skip-gram fashion) simultaneous with DBOW doc-vector training; If 0, only trains doc-vectors (faster).
		  context_size=10,
		  lr=0.005,
		  epochs=100,
		  negative_sampling=10
		  ):
	gensimLoader = models_gensim.GensimModels()
	preprocess = Preprocessor()

	"""
	Mode Description

	# mode 1 : Embed Ingredients with Chemical Compounds
	# mode 999 : Plot Loaded Word2Vec or Doc2vec
	"""
	if mode == 'train':
		"""
		Load Data
		"""
		path_data = "../data/"
		path_file = "flavordb_ver2.0.csv"
		path_load = path_data + path_file
		path_save = '../models/'+ 'file-' + str(path_file) + '_model_ver-' + str(model_ver) + '.bin'

		df = pd.read_csv(path_load)
		"""
		Preprocess Data

		"""
		# build taggedDocument form of corpus
		corpus_ingr2vec = list(preprocess.build_taggedDocument_df(df, random_sampling, num_sampling, context_size))

		"""
		Build & Save Doc2Vec
		"""
		# build ingredient embeddings with doc2vec
		model_ingr2vec = gensimLoader.build_doc2vec(corpus_ingr2vec,
													load_pretrained,
													path_pretrained,
													doc_dim,
													word_dim,
													dm,
													dm_mean,
													dm_concat,
													dbow_words,
													context_size,
													lr,
													epochs,
													negative_sampling)


		# save character-level compounds embeddings with doc2vec
		gensimLoader.save_doc2vec_only_doc(model=model_ingr2vec, path=path_save)
		model_loaded = gensimLoader.load_word2vec(path=path_save)

		print(len(model_loaded.vocab))
		#for x in model_loaded.vocab:
		#	print(x, len(model_loaded.word_vec(x)))

	elif mode == 'plot':
		"""
		Plot Ingredient2Vec

		"""
		path_save = '../models/'+ 'file-' + str(path_file) + '_model_ver-' + str(model_ver) + '.bin'
		model_loaded = gensimLoader.load_word2vec(path=path_save)
		ingr2vec = {}
		for x in model_loaded.vocab:
			ingr2vec[x] = model_loaded.word_vec(x)
		ingr2vec_tsne = plot.load_TSNE(ingr2vec, dim=2)

		ingr2cate = {}
		df_info = pd.read_csv("../data/flavordb/D2_fdb_info.csv", sep=",")
		for ingr in ingr2vec:
			df_info['ingredient_name'] = df_info['ingredient_name'].str.lower()
			try:
				cate = df_info[(df_info['ingredient_name']==ingr)]['ingredient_category'].values[0]
				ingr2cate[ingr] = cate
			except IndexError:
				ingr2cate[ingr] = 'None'

		print("# of Ingredients in FlavorDB: {}".format(len(ingr2cate)))
		ingr2cate = preprocess.contract_category(ingr2cate)

		plot.plot_category(ingr2vec, ingr2vec_tsne, path_save.replace("bin", "html"), ingr2cate, withLegends=True)
		#DataPlotter.plot_clustering(model_loaded, model_tsne, Config.path_plottings_ingredients_clustering)

	else:
		print("Please specify the mode you want.")

if __name__ == '__main__':
	fire.Fire()
