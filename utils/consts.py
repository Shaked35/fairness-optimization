import os.path

GENRES = ['Sci-Fi', 'Musical', 'Action', 'Crime', 'Romance']
WOMEN_GENRES = ['Musical', 'Romance']
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
NUMBER_OF_SYNTHETIC_TESTS = 10
ML_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
RESOURCE_DIR = "resource"
DATASET_NAME = "ml-1m"
ZIP_NAME = os.path.join(RESOURCE_DIR, 'ml-1m.zip')
LOCAL_DATASET_LOCATION = os.path.join(RESOURCE_DIR, DATASET_NAME)
# W = women who do enjoy STEM topics
# WS = women who do not enjoy STEM topics
# M = men who do enjoy STEM topics
# MS = men who do not enjoy STEM topics
# Masc = courses that tend to appeal to men
# Fem = courses that tend to appeal to women
USER_GROUPS = ['W', 'WS', 'M', 'MS']
ITEM_GROUPS = ['Fem', 'STEM', 'Masc']
NUM_USERS = 400
NUM_ITEMS = 300
BATCH_SIZE = 256
EPOCHS = 200
N_EPOCHS_PATIENCE = 6
WORKERS = 50
MAX_QUEIE_SIZE = 5
NUMBER_OF_NEGATIVE_EXAMPLES = 4
TOP_K = 10
MIN_DELTA = 1e-6
DEFAULT_LAMBDA = 0.001
EPOCH_PRINT = 10
