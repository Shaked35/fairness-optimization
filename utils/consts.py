import os.path

GENRES = ['Action', 'Crime', 'Musical', 'Romance', 'Sci-Fi']
RATING_COLUMNS = ['user_id', 'movie_id', 'rating', 'timestamp']
DATA_COLS = ['user_id', 'item_id', 'rating', 'timestamp']
USER_COLS = ['user_id', 'age', 'gender', 'occupation', 'zip code']
ITEM_COLS = ['movie id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action',
             'Adventure', 'Animation',
             'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film_Noir', 'Horror', 'Musical',
             'Mystery', 'Romance',
             'Sci_Fi', 'Thriller', 'War', 'Western']
DELIMITER = '|'
MIN_USER_RATED = 50
NUMBER_OF_TESTS = 5
ML_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
RESOURCE_DIR = "resource"
DATASET_NAME = "ml-1m"
ZIP_NAME = os.path.join(RESOURCE_DIR, 'ml-1m.zip')
LOCAL_DATASET_LOCATION = os.path.join(RESOURCE_DIR, DATASET_NAME, DATASET_NAME)