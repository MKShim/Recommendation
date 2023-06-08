import requests
import pandas as pd
import sys
from tqdm import tqdm
import time

def add_url(row):
    return f"http://www.imdb.com/title/tt{row}/"


def add_rating(df):
    ratings_df = pd.read_csv('app/data/ratings.csv')
    ratings_df['movieId'] = ratings_df['movieId'].astype(str) 
    agg_df = ratings_df.groupby('movieId').agg(
        rating_count=('rating', 'count'),
        rating_avg=('rating', 'mean') 
        ).reset_index()
    rating_added_df = df.merge(agg_df, on='movieId') 
    return rating_added_df


def add_poster(df):
    for i, row in tqdm(df.iterrows(),total=df.shape[0]):
        tmdb_id = row['tmdbId']
        tmdb_url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key=ebef0b02bc720e588b566f2cf41027b4&language-en-US"
        result = requests.get(tmdb_url)
        try:
            df.at[i, "poster_path"] = "https://image.tmdb.org/t/p/original" + result.json()['poster_path']
            time.sleep(0.1) # 0.1sec
        except (TypeError, KeyError) as e: # toy story poster as default
            df.at[i, "poster_path"] = "https://image.tmdb.org/t/p/original/uXDFjJbaP4jW5hWSBrPrlKpxab.jpg"
    return df


if __name__=="__main__":
    movies_df = pd.read_csv('app/data/movies.csv')
    #print(movies_df)
    movies_df['movieId'] = movies_df['movieId'].astype(str)
    links_df = pd.read_csv ('app/data/links.csv', dtype=str)
    merged_df = movies_df.merge(links_df, on= 'movieId', how='left')
    merged_df['url'] = merged_df['imdbId'].apply(lambda x: add_url (x))
    result_df= add_rating(merged_df)
    result_df['poster_path']=None
    result_df= add_poster(result_df)

    result_df.to_csv('app/data/movies_final.csv', index=False)
    #print(result_df)
    
