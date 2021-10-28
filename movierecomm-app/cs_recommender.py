import random
#from nmf_model import get_model, ratings, features_movies
from cs_model import ratings

import pandas as pd
import numpy as np
#from sklearn.impute import KNNImputer
#from sklearn.decomposition import NMF

def get_recommendation(form_data):
    
    ratings_app = ratings.copy()

    # initialize new user  
    #new_user = np.zeros(shape= (1,len(ratings_app.columns))).tolist()[0] # <-- list of zeros
    new_user = [0] * len(ratings_app.columns) # <-- list of zeros
    #new_user

    # get new user ratings
    new_user_ratings = form_data
    #new_user_ratings

    for index, item in enumerate(ratings_app.columns):
        #print(index,item)
        if new_user_ratings.get(item): # <-- Return the value for key if key is in the dictionary
            #print(index, item)
            new_user[index] = new_user_ratings[item]

    #new_user

    #ratings_app.index[-1]
    #len(ratings_app)

    # new user dataframe
    new_user_df = pd.DataFrame([new_user],index = [ratings_app.index[-1]+1],columns= ratings_app.columns)
    #new_user_df

    #concat the new user to the ratings df
    ratings_app = pd.concat([ratings_app, new_user_df])
    #ratings_app.tail(1)

    # use the transposed version of ratings
    ratings_app_T = ratings_app.T
    #ratings_app_T.head(1)

    ## Model

    from sklearn.metrics.pairwise import cosine_similarity
    cs = cosine_similarity(ratings_app)
    #With this matrix, we can create recommendations

    #pd.DataFrame(cs).index[-1]

    # choose an active user, in this case the new added user 
    active_user = pd.DataFrame(cs).index[-1]

    # create a list of unseen movies for this user
    unseen_movie = list(ratings_app_T.index[ratings_app_T[active_user] == 0])
    #unseen_movie

    # Create a list of top 3 similar user (nearest neighbours)
    neighbours = list(pd.DataFrame(cs)[active_user].sort_values(ascending=False).index[1:4])
    #neighbours

    # create the recommendation (predicted/rated movie)
    predicted_ratings_movies = []
    for movie in unseen_movie:
        
        # we check the users who watched the movie
        others = list(ratings_app_T.columns[ratings_app_T.loc[movie].values != 0])
        #print(others)
        num = 0
        den = 0
        for user in neighbours:
            if user in others:
            #  we want to extract the ratings and similarities
                rating = ratings_app_T.loc[movie,user]
                #print(rating)
                similarity = pd.DataFrame(cs).loc[active_user,user]
                #print('similarity: ', similarity)

                
            # predict the rating based on the (weighted) average ratings of the neighbours
            # sum(ratings)/no.users OR 
            # sum(ratings*similarity)/sum(similarities)
                num += rating*similarity
                #print('num: ', num)
                den += similarity
                #print('den: ', den)
        if den!= 0:
            predicted_ratings = num/den
            predicted_ratings_movies.append([predicted_ratings,movie]) 

    #predicted_ratings_movies

    # create df pred
    df_pred = pd.DataFrame(predicted_ratings_movies,columns = ['rating','movie'])
    return list(df_pred.sort_values(by=['rating'],ascending=False)['movie'].head(5))

