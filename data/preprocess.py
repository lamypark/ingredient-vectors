import json
import pandas as pd
import numpy as np
import pickle
import os
from os import listdir
from os.path import isfile, join
import random

df_ingredients_info = pd.read_csv("./flavordb/D2_fdb_info.csv", sep=",")
path_compounds_info = './flavordb/compounds_info.pkl'
save_path = './flavordb_ver3.0.csv'

id2compound = './flavordb/id2compound.pkl'
compound2id = './flavordb/compound2id.pkl'

with open(path_compounds_info, 'rb') as pkl:
    dict_compounds_info = pickle.load(pkl)

f_write = open(save_path, 'w')
f_write.write("text,label\n")

print(len(df_ingredients_info))
for index, row in df_ingredients_info.iterrows():
    ingredient = row['ingredient_name']
    compounds_ids = row['compound_ids'][1:-1].split(",")
    compounds_ids_clean = []

    for id in compounds_ids:
        compounds_ids_clean.append(id.strip())

    if len(compounds_ids_clean) > 10:
        for i in range(50):
            sampled_compounds_ids = random.sample(compounds_ids_clean, 10)
            #print(sampled_compounds_ids)
            sampled_compounds_ids = " ".join(sampled_compounds_ids)
            f = "\""+ sampled_compounds_ids + "\"," + ingredient
            #print(f)
            f_write.write(f)
            f_write.write("\n")



f_write.close()
