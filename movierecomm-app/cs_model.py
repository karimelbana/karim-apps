#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

#get data

movies = pd.read_csv('./Data_small/movies.csv')
ratings_data = pd.read_csv('./Data_small/ratings.csv')
movies_rating = pd.merge(movies, ratings_data, on='movieId')

df = pd.pivot_table(movies_rating, values='rating', index='userId',
                    columns=['title'])

# # replacing nans with zeros

ratings = df.fillna(value=0.0)

