
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import seaborn as sns
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')
import base64
import io
from matplotlib.pyplot import imread
import codecs
from IPython.display import HTML
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

movies = pd.read_csv('movies_5000.csv')
credits = pd.read_csv('credits_5000.csv')

movies['genres'] = movies['genres'].apply(json.loads)
for index,i in zip(movies.index,movies['genres']):
    list1 = []
    for j in range(len(i)):
        list1.append((i[j]['name']))
    movies.loc[index,'genres'] = str(list1)

movies['keywords'] = movies['keywords'].apply(json.loads)
for index,i in zip(movies.index,movies['keywords']):
    list1 = []
    for j in range(len(i)):
        list1.append((i[j]['name']))
    movies.loc[index,'keywords'] = str(list1)

movies['production_companies'] = movies['production_companies'].apply(json.loads)
for index,i in zip(movies.index,movies['production_companies']):
    list1 = []
    for j in range(len(i)):
        list1.append((i[j]['name']))
    movies.loc[index,'production_companies'] = str(list1)

credits['cast'] = credits['cast'].apply(json.loads)
for index,i in zip(credits.index,credits['cast']):
    list1 = []
    for j in range(len(i)):
        list1.append((i[j]['name']))
    credits.loc[index,'cast'] = str(list1)

credits['crew'] = credits['crew'].apply(json.loads)
def director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
credits['crew'] = credits['crew'].apply(director)
credits.rename(columns={'crew':'director'},inplace=True)

movies = movies.merge(credits,left_on='id',right_on='movie_id',how='left')
movies = movies[['id','original_title','overview','genres','cast','vote_average','director','keywords']]

movies['genres'] = movies['genres'].str.strip('[]').str.replace(' ','').str.replace("'",'')
movies['genres'] = movies['genres'].str.split(',')

list1 = []
for i in movies['genres']:
    list1.extend(i)

for i,j in zip(movies['genres'],movies.index):
    list2=[]
    list2=i
    list2.sort()
    movies.loc[j,'genres']=str(list2)
movies['genres'] = movies['genres'].str.strip('[]').str.replace(' ','').str.replace("'",'')
movies['genres'] = movies['genres'].str.split(',')

genreList = []
top_genres = pd.Series(list1).value_counts()[:20].index.tolist()

for index, row in movies.iterrows():
    genres = row["genres"]

    for genre in genres:
        if genre in top_genres:
            genreList.append(genre)

genreList = list(set(genreList))
#print(genreList)

def binary(genre_list):
    binaryList = []
    
    for genre in genreList:
        if genre in genre_list:
            binaryList.append(1)
        else:
            binaryList.append(0)
    
    return binaryList

movies['genres_bin'] = movies['genres'].apply(lambda x: binary(x))
movies['genres_bin'].head()

movies['cast'] = movies['cast'].str.strip('[]').str.replace(' ','').str.replace("'",'').str.replace('"','')
movies['cast'] = movies['cast'].str.split(',')

list1=[]
for i in movies['cast']:
    list1.extend(i)

for i,j in zip(movies['cast'],movies.index):
    list2 = []
    list2 = i[:4]
    movies.loc[j,'cast'] = str(list2)
movies['cast'] = movies['cast'].str.strip('[]').str.replace(' ','').str.replace("'",'')
movies['cast'] = movies['cast'].str.split(',')
for i,j in zip(movies['cast'],movies.index):
    list2 = []
    list2 = i
    list2.sort()
    movies.loc[j,'cast'] = str(list2)
movies['cast']=movies['cast'].str.strip('[]').str.replace(' ','').str.replace("'",'')

castList = []
top_casts = pd.Series(list1).value_counts().drop('').index.tolist()[:20]
top_casts = [actor.replace('Jr.', 'Robert Downey Jr.') for actor in top_casts]
#print(top_casts)

for index, row in movies.iterrows():
    cast = row["cast"]

    for i in cast:
        if i in top_casts:
            castList.append(i)

def binary(cast_list):
    binaryList = []

    for actor in top_casts:
        if actor in cast_list:
            binaryList.append(1)
        else:
            binaryList.append(0)

    return binaryList

movies['cast_bin'] = movies['cast'].apply(lambda x: binary(x))
movies['cast_bin'].head()

def xstr(s):
    if s is None:
        return ''
    return str(s)
movies['director'] = movies['director'].apply(xstr)

list1=[]
for i in movies['director']:
    list1.extend(i)

top_directors = movies['director'].explode().value_counts().drop('').index[:20].tolist()
directorList = []
#print(top_directors)
for index, row in movies.iterrows():
    cast = row["director"]
    for i in cast:
        if i in top_directors:
            directorList.append(i)

def binary(director_list):
    binaryList = []
    for direct in top_directors:
        if direct in director_list:
            binaryList.append(1)
        else:
            binaryList.append(0)
    return binaryList

movies['director_bin'] = movies['director'].apply(lambda x: binary(x))
movies.head()

stop_words = set(stopwords.words('english'))
stop_words.update(',',';','!','?','.','(',')','$','#','+',':','...',' ','')

words=movies['keywords'].dropna().apply(nltk.word_tokenize)
word=[]
for i in words:
    word.extend(i)
word=pd.Series(word)
word=([i for i in word.str.lower() if i not in stop_words])

movies['keywords'] = movies['keywords'].str.strip('[]').str.replace(' ','').str.replace("'",'').str.replace('"','')
movies['keywords'] = movies['keywords'].str.split(',')
for i,j in zip(movies['keywords'],movies.index):
    list2 = []
    list2 = i
    movies.loc[j,'keywords'] = str(list2)
movies['keywords'] = movies['keywords'].str.strip('[]').str.replace(' ','').str.replace("'",'')
movies['keywords'] = movies['keywords'].str.split(',')
for i,j in zip(movies['keywords'],movies.index):
    list2 = []
    list2 = i
    list2.sort()
    movies.loc[j,'keywords'] = str(list2)
movies['keywords'] = movies['keywords'].str.strip('[]').str.replace(' ','').str.replace("'",'')
movies['keywords'] = movies['keywords'].str.split(',')

top_words = movies['keywords'].explode().value_counts().drop('').index[:20].tolist()
words_list = []
#print(top_words)
for index, row in movies.iterrows():
    keywords = row["keywords"]
    
    for keyword in keywords:
        if keyword in top_words:
            words_list.append(keywords)

def binary(words):
    binaryList = []
    for keyword in top_words:
        if keyword in words_list:
            binaryList.append(1)
        else:
            binaryList.append(0)
    return binaryList

movies['words_bin'] = movies['keywords'].apply(lambda x: binary(x))
movies = movies[(movies['vote_average']!=0)]
movies = movies[movies['director']!='']

def Soft_Set(row):
    genA = row['genres_bin']
    scorA = row['cast_bin']
    directA = row['director_bin']
    wordsA = row['words_bin']
    return [genA, scorA, directA, wordsA]
if 'Soft-Set' not in movies.columns:
    movies['Soft-Set'] = movies.apply(Soft_Set, axis=1)

def find_most_similar_film(query_string):
    distances = []

    query_film = movies.loc[movies['original_title'] == query_string, 'Soft-Set'].values[0]
    for _, row in movies.iterrows():
        soft_set = row['Soft-Set']
        distance = np.sqrt(np.sum(np.square(np.array(soft_set) - np.array(query_film))))
        distances.append(distance)

    nearest_indices = np.argsort(distances)[0:8]
    similar_films = movies.loc[nearest_indices, 'original_title'].tolist()

    return similar_films
def find_most_similar_desc(query_string):
    distances = []

    query_film = movies.loc[movies['original_title'] == query_string, 'Soft-Set'].values[0]
    for _, row in movies.iterrows():
        soft_set = row['Soft-Set']
        distance = np.sqrt(np.sum(np.square(np.array(soft_set) - np.array(query_film))))
        distances.append(distance)

    nearest_indices = np.argsort(distances)[0:8]
    similar_desc = movies.loc[nearest_indices, 'overview'].tolist()
    
    return similar_desc

# print(find_most_similar_film(query_film))

# def cosine_similarity(vector1, vector2):
#     dot_product = np.dot(vector1, vector2)
#     norm1 = np.linalg.norm(vector1)
#     norm2 = np.linalg.norm(vector2)
    
#     similarity = dot_product / (norm1 * norm2)
    
#     return similarity
