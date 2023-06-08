# endpoint : /all 
import pandas as pd

item_fname = '/Users/minkyoung/Documents/Dev/recom/app/data/movies_final.csv'

def random_items():
    movies_df = pd.read_csv(item_fname)
    movies_df = movies_df.fillna('') 
    result_items = movies_df.sample(n=10).to_dict("records") 
    return result_items


def random_genres_items(genre):
    movies_df = pd.read_csv(item_fname)
    genre_df = movies_df[movies_df['genres'].apply(lambda x: genre in x.lower())]
    genre_df = movies_df.fillna('')
    result_items = genre_df.sample(n=10).to_dict("records")
    return result_items

