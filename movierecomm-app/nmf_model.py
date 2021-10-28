#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.decomposition import NMF

#get data

movies = pd.read_csv('../Data_small/movies.csv')
ratings_data = pd.read_csv('../Data_small/ratings.csv')
movies_rating = pd.merge(movies, ratings_data, on='movieId')

df = pd.pivot_table(movies_rating, values='rating', index='userId',
                    columns=['title'])


# # KNN Imputing

imputer = KNNImputer(n_neighbors=2)

ratings = pd.DataFrame(imputer.fit_transform(df), 
                       index = df.index, 
                       columns = df.columns)

# # Model

#1 create movies features (genres for ex)
nmf = NMF(n_components=2)
nmf.fit(ratings)

#nmf.components_

# # Save the model

import pickle

#save model
binary = pickle.dumps(nmf)
open('nmf_model.bin', 'wb').write(binary)

# 2 create features-movies matrix

movies = ratings.columns.values.tolist()

features_movies = pd.DataFrame(nmf.components_, 
                 columns=movies, 
                 index=['feature1', 'feature2'])
#features_movies.head(5)

users = ratings.index.values.tolist()

# 3 create users-features matrix

users_features = pd.DataFrame(nmf.transform(ratings), 
                 columns=['feature1', 'feature2'], 
                 index=users)
#users_features.head(5)

#features_movies.shape

#users_features.shape

# 4 build ratings recommender matrix

users_movies_recomm = pd.DataFrame(np.dot(users_features, features_movies), 
                                  index=users, 
                                  columns=movies)
print('\nRatings Recommender Matrix:\n')
print(
    users_movies_recomm.head(5)
)

# # Add a new user

#UserX = [[NaN, NaN, 0, 0, 5, 3, 2 ,1]] # Ratings the user gives for some movies not for the whole list of movies we have
#here roll over list of movies rated by the user and add their entires, everything else should be nans
#UserX_features = nmf.transform(UserX)
#UserX_movies_recomm = np.dot(UserX_features, features_movies) #features_movies does not change as we didn not add any new movies
#UserX_movies_recomm = pd.DataFrame(UserX_movies_recomm, index=['UserX'], columns=movies)


# # Add a new movie

#New_Movie = pd.DataFrame({'New_Movie':[0, 0, 4, 0, 0, 4, 5, 5, 5]}, index = ratings.index)
#New_Movie

#ratings_w_New_Movie = ratings.merge(New_Movie, right_index=True, left_index=True)
#ratings_w_New_Movie

#users.append('UserX')
#movies.append('New_Movie')

#rebuilding the model after adding a new movie
#1 create movies features (genres for ex)

#nmf_new = NMF(n_components=2)
#nmf_new.fit(ratings_w_New_Movie)

# 2 create features-movies matrix

#features_movies_new=pd.DataFrame(nmf_new.components_, 
                 #index=['feature1', 'feature2'], 
                 #columns=movies)
#features_movies_new.head(5)

# 3 create users-features matrix

#users_features_new = nmf_moonlight.transform(ratings_w_New_Movie)
#users_features_new.head(5)

# 4 build ratings recommender matrix

#users_movies_recomm_new = pd.DataFrame(np.dot(users_features_new, features_movies_new), index=users, columns=movies)
#users_movies_recomm_new.head(5)

# # Save the model

#import pickle

#save model
#binary = pickle.dumps(nmf)
#open('nmf_model.bin', 'wb').write(binary)

#read model
#binary = open('nmf_model.bin', 'rb').read()
#nmf_read = pickle.loads(binary)

#nmf_read.components_

def get_model():
    #read model
    binary = open('nmf_model.bin', 'rb').read()
    nmf_read = pickle.loads(binary)
    return nmf_read
