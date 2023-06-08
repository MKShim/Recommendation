import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares
import pickle

saved_model_fname="/Users/minkyoung/Documents/Dev/recom/app/model/finalized_model.sav"   # model CHK
data_fname= "/Users/minkyoung/Documents/Dev/recom/app/data/ratings.csv"
item_fname= "/Users/minkyoung/Documents/Dev/recom/app/data/movies_final.csv"
weight = 10

def model_train():
    ratings_df = pd.read_csv(data_fname)
    
    ratings_df["movieId"] = ratings_df["movieId"].astype("category") # row
    ratings_df["userId"] = ratings_df["userId"].astype("category")   # col

    # create a sparse matrix of all the users/repos : (9724 x 610) matrix, elements are rating values.
    rating_matrix = coo_matrix(
        (
            ratings_df["rating"].astype(np.float32),                # element values
            (
                ratings_df["userId"].cat.codes.copy(),              # row : 610
                ratings_df["movieId"].cat.codes.copy(),             # col : 9724
                                          
            )
        ) #shape=(9724, 610)
    )

    als_model = AlternatingLeastSquares(                                    # factorized matrix, ALS model
        factors=50, regularization=0.01, dtype=np.float32, iterations=50
    )

    als_model.fit(weight * rating_matrix)

    pickle.dump(als_model, open(saved_model_fname, "wb"))
    return als_model


def calculate_item_based(mapped_idx, movieid_dict):
    loaded_model = pickle.load(open(saved_model_fname, "rb"))
    recs = loaded_model.similar_items(itemid=int(mapped_idx), N=11)         # Using ALS model, pick 11 movies 
    return [str(movieid_dict[r]) for r in recs[0]]


#  getting all information(including title, genres,... )
def item_based_recommendation(movie_id):             
    ratings_df = pd.read_csv(data_fname)
    ratings_df["movieId"] = ratings_df["movieId"].astype("category")
    ratings_df["userId"] = ratings_df["userId"].astype("category")
    movies_df = pd.read_csv(item_fname)

    movieid_dict = dict(enumerate(ratings_df["movieId"].cat.categories))               # {0 ~ int : movieID}
    try:
        mapped_idx = ratings_df["movieId"].cat.categories.get_loc(int(movie_id))        # 0 ~ int, return mapping index corresponding with movieID
        result = calculate_item_based(mapped_idx, movieid_dict)
    except KeyError as e:
        result = []
    result = [int(x) for x in result if x != movie_id]
    result_items = movies_df[movies_df["movieId"].isin(result)].to_dict("records")
    return result_items


def build_matrix_input(input_rating_dict, movieID_dict):
    model = pickle.load(open(saved_model_fname, "rb"))
    # input rating list dictionary : {movie_id : ratings}, ex {1: 4.0, 2: 3.5, 3: 5.0}

    movie_ids = {r: i for i, r in movieID_dict.items()}                              # { movieID : 0 ~ int }
    mapped_idx = [movie_ids[s] for s in input_rating_dict.keys() if s in movie_ids] # input - movieID, return index keys of selected movieID    
    data = [weight * float(x) for x in input_rating_dict.values()]                  # input - star

    # print('mapped index', mapped_idx)
    # print('weight data', data)
    rows = [0 for _ in mapped_idx]                                                  # put 0, number of index keys, selected movies
    #shape = (1, len(mapped_idx))
    shape = (1, model.item_factors.shape[0])  # print shape : {"result":[1,610]}    # shape[0] : the number of row, the number of user = 1
    return coo_matrix((data, (rows, mapped_idx)), shape=shape).tocsr()
    #return shape


def calculate_user_based(sparse_user_item, movieID_dict):
    loaded_model = pickle.load(open(saved_model_fname, "rb"))
    recommended = loaded_model.recommend(
        userid=0, user_items=sparse_user_item, recalculate_user=True, N=10
    )
    #return [str(items[r]) for r, s in recommended]
    return [str(movieID_dict[r]) for r in recommended[0]]


def user_based_recommendation(input_ratings):
    ratings_df = pd.read_csv(data_fname)
    ratings_df["movieId"] = ratings_df["movieId"].astype("category")           # row
    ratings_df["userId"] = ratings_df["userId"].astype("category")             # col
    movies_df = pd.read_csv(item_fname)

    movieID_dict = dict(enumerate(ratings_df["movieId"].cat.categories))       # {0 ~ int : movieID}
    coo_matrix = build_matrix_input(input_ratings, movieID_dict)             #
    result = calculate_user_based(coo_matrix, movieID_dict)
    result = [int(x) for x in result]
    
    result_items = movies_df[movies_df["movieId"].isin(result)].to_dict("records")
    return result_items
    #return coo_matrix

if __name__=="__main__":
    model = model_train()

