import random
from nmf_model import get_model, ratings, features_movies

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.decomposition import NMF


def get_recommendation(form_data):
    
    ratings_app = ratings.copy()

    nmf = get_model()
    
    # # Add a new user
    UserY=[]
    for key in ratings_app.columns[0:5]:
        if key in form_data.keys():
            #print(form_data[key])
            UserY.append(form_data[key])
    #print('New User Received Ratings:\n', UserY)
    
    UserY_nan=[np.nan] * (len(ratings_app.columns)-len(ratings_app.columns[0:5]))
    #len(UserY_nan)
    UserY = [UserY + UserY_nan]
    #len(UserY)

    ratings_app = pd.concat([ratings_app, 
        pd.DataFrame(UserY, columns=ratings_app.columns, index={ratings_app.index[-1]+1})])
    
    #impute to get rid of nans
    imputer = KNNImputer(n_neighbors=2)
    ratings_app = pd.DataFrame(imputer.fit_transform(ratings_app), 
        columns = ratings_app.columns, index = ratings_app.index)

    user_features = nmf.transform(ratings_app)
    movies_recomm = np.dot(user_features, features_movies) #features_movies does not change as we didn not add any new movies
    #print('Recommendations:\n', pd.DataFrame(movies_recomm, columns = ratings.columns, index = ratings.index).Iloc[-1, ])
    
    return pd.DataFrame(movies_recomm, columns = ratings_app.columns, index = ratings_app.index).iloc[-1, ].sort_values(
        ascending=False).head(5).index



#UserX = [[3, 2 ,1]] # Ratings the user gives for some movies not for the whole list of movies we have
#here roll over list of movies rated by the user and add their entires, everything else should be nans
#UserX_features = nmf.transform(UserX)
#UserX_movies_recomm = np.dot(UserX_features, features_movies) #features_movies does not change as we didn not add any new movies
#UserX_movies_recomm = pd.DataFrame(UserX_movies_recomm, index=['UserX'], columns=movies)


#return UserX_movies_recomm

