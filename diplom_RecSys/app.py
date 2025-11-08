from flask import Flask, request, jsonify
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Загрузка модели
print("Загружаем модель...")
with open('models/svd_model.pkl', 'rb') as f_in:
    model = pickle.load(f_in)

# Загрузка данных
print("Загружаем данные...")
movies = pd.read_csv('data/u.item', sep='|', encoding='latin-1',
                    names=['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL',
                           'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy',
                           'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                           'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
movies = movies[['movie_id', 'movie_title']].rename(columns={'movie_id': 'item_id', 'movie_title': 'title'})

# Загрузка trainset (упрощенная версия)
print("Подготавливаем данные...")
from surprise import Dataset, Reader
import pandas as pd

# Загружаем данные для trainset
u1_base = pd.read_csv('data/u1.base', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
reader = Reader(rating_scale=(1, 5))
dataset = Dataset.load_from_df(u1_base[['user_id', 'item_id', 'rating']], reader)
trainset = dataset.build_full_trainset()

def get_top_n_recommendations(algo, user_id, trainset, n=10):
     all_items = set(trainset.all_items())
    try:
        inner_uid = trainset.to_inner_uid(user_id)
        seen = {j for (j, _) in trainset.ur[inner_uid]}
    except ValueError:
        seen = set()  # Пользователь не найден, считаем, что фильмы не были просмотрены
        # Извлекаем все сырые рейтинги из trainset
        ratings = [(trainset.to_raw_iid(iid), rating) for (uid, iid, rating) in trainset.all_ratings()]
        ratings_df = pd.DataFrame(ratings, columns=['item_id', 'rating'])
        # Получаем средний рейтинг на item_id
        ratings_agg = ratings_df.groupby('item_id', as_index=False)['rating'].mean()
        merged = movies.merge(ratings_agg, on='item_id', how='left')
        popular = merged.sort_values(by='rating', ascending=False).head(n)
        return popular[['title', 'rating']].to_dict(orient='records')

    predictions = []
    for item_id in all_items - seen:
        pred = algo.predict(user_id, trainset.to_raw_iid(item_id))
        predictions.append((trainset.to_raw_iid(item_id), pred.est))
    predictions.sort(key=lambda x: x[1], reverse=True)

    top_df = pd.DataFrame(predictions[:n], columns=['item_id', 'predicted_rating'])
    top_df = top_df.merge(movies, on='item_id', how='left')
    return top_df[['title', 'predicted_rating']].to_dict(orient='records')

@app.route('/')
def home():
    return '''
    <h1>Рекомендательная система фильмов</h1>
    <p>Используйте /recommend?user_id=ВАШ_ID</p>
    <p>ID от 1 до 943 - персонализированные рекомендации</p>
    <p>Любой другой ID - рекомендации по популярности</p>
    '''

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id', type=int)
    if not user_id:
        return jsonify({"error": "Укажите user_id"})
    
    recommendations = get_top_n_recommendations(model, user_id, trainset, n=10)
    return jsonify(recommendations)

if __name__ == '__main__':
    print("Сервер запускается...")
    app.run(host='0.0.0.0', port=5000, debug=True)
