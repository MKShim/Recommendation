from typing import List, Optional
from fastapi import FastAPI, Query
from recommender import item_based_recommendation, user_based_recommendation
from resolver import random_items, random_genres_items
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello"}


@app.get("/all/")
async def all_movies():
    result = random_items()
    return {"result": result}

@app.get("/genres/{genre}")
async def genre_movies(genre: str):
    result = random_genres_items(genre)
    return {"result": result}

# endpoint for recommended items
@app.get("/item-based/{movie_id}")
async def item_based(movie_id: str):
    result = item_based_recommendation(movie_id)
    return {"result": result}

@app.get("/user-based/")    
#(query parameter)
#async def user_based(idnmoive: List[str]):
async def user_based(idnmoive: Optional[List[str]] = Query(None)):
    input_ratings_dict = dict(
        (int(x.split(":")[0]), float(x.split(":")[1])) for x in idnmoive
    )
    result = user_based_recommendation(input_ratings_dict)
    return {"result": result}



