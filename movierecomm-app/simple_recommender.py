import random
from model_V2 import get_model

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.decomposition import NMF


def get_recommendation(form_data):
    
    ratings, nmf, features_movies = get_model()

    # # Add a new user
    UserX=[[form_data['Titanic'], np.nan, np.nan, form_data['Star Trek'], form_data['Star Wars']]]
    print('New User Received Ratings:\n', UserX)
    
    ratings = pd.concat([ratings, pd.DataFrame(UserX, columns=ratings.columns, index={'UserX'})])
    
    #impute to get rid of nans
    imputer = KNNImputer(n_neighbors=2)
    ratings = pd.DataFrame(imputer.fit_transform(ratings), 
        columns = ratings.columns, index = ratings.index)

    user_features = nmf.transform(ratings)
    movies_recomm = np.dot(user_features, features_movies) #features_movies does not change as we didn not add any new movies
    #print('Recommendations:\n', pd.DataFrame(movies_recomm, columns = ratings.columns, index = ratings.index))
    
    return pd.DataFrame(movies_recomm, columns = ratings.columns, index = ratings.index).loc['UserX', ]



#UserX = [[3, 2 ,1]] # Ratings the user gives for some movies not for the whole list of movies we have
#here roll over list of movies rated by the user and add their entires, everything else should be nans
#UserX_features = nmf.transform(UserX)
#UserX_movies_recomm = np.dot(UserX_features, features_movies) #features_movies does not change as we didn not add any new movies
#UserX_movies_recomm = pd.DataFrame(UserX_movies_recomm, index=['UserX'], columns=movies)


#return UserX_movies_recomm

