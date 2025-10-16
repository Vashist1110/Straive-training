import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt

np.random.seed(42)

# === 1. Generate Synthetic Data ===

# Users
n_users = 1000
user_ids = np.arange(n_users)
ages = np.random.randint(18, 70, n_users)
genders = np.random.choice(['M', 'F'], n_users)
user_data = pd.DataFrame({'user_id': user_ids, 'age': ages, 'gender': genders})

# Movies
n_movies = 500
movie_ids = np.arange(n_movies)
genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Sci-Fi']
movie_genres = np.random.choice(genres, n_movies)
durations = np.random.randint(80, 180, n_movies)
release_years = np.random.randint(1980, 2023, n_movies)
movie_ratings = np.round(np.random.uniform(1, 10, n_movies), 1)
movie_data = pd.DataFrame({
    'movie_id': movie_ids,
    'genre': movie_genres,
    'duration': durations,
    'release_year': release_years,
    'rating': movie_ratings
})

# Interactions (user watched movie + rating + liked)
n_interactions = 10000
interaction_user_ids = np.random.choice(user_ids, n_interactions)
interaction_movie_ids = np.random.choice(movie_ids, n_interactions)
user_ratings = np.round(np.random.uniform(1, 10, n_interactions), 1)
liked = (user_ratings >= 6).astype(int)  # liked if rating >= 6
watch_times = np.random.choice(['morning', 'afternoon', 'evening', 'night'], n_interactions)

interaction_data = pd.DataFrame({
    'user_id': interaction_user_ids,
    'movie_id': interaction_movie_ids,
    'user_rating': user_ratings,
    'liked': liked,
    'watch_time': watch_times
})

# Merge interaction with user and movie for preprocessing
data = interaction_data.merge(user_data, on='user_id', how='left')
data = data.merge(movie_data, on='movie_id', how='left')

# === 2. Data Preparation ===

# Handle missing values (none in synthetic data, but good practice)
data.fillna({'age': data['age'].median(),
             'gender': 'M',
             'genre': 'Drama',
             'duration': data['duration'].median(),
             'release_year': data['release_year'].median(),
             'rating': data['rating'].median()}, inplace=True)

# Encode categorical variables
# Gender to binary
data['gender'] = data['gender'].map({'M': 0, 'F': 1})

# One-hot encode genre
genre_dummies = pd.get_dummies(data['genre'], prefix='genre')
data = pd.concat([data, genre_dummies], axis=1)

# One-hot encode watch_time
watch_dummies = pd.get_dummies(data['watch_time'], prefix='watch')
data = pd.concat([data, watch_dummies], axis=1)

# === 3. Data Validation / Split ===

# Split so users in test are also in train
train_val_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['liked'])

# Ensure users in test appear in train_val
train_val_users = train_val_data['user_id'].unique()
test_data = test_data[test_data['user_id'].isin(train_val_users)]

# Now split train and val
train_data, val_data = train_test_split(train_val_data, test_size=0.2, random_state=42,
                                        stratify=train_val_data['liked'])

print(f"Train size: {train_data.shape}, Val size: {val_data.shape}, Test size: {test_data.shape}")

# === 4. Feature Engineering ===

# User-level features
user_avg_rating = train_data.groupby('user_id')['user_rating'].mean().reset_index().rename(
    columns={'user_rating': 'user_avg_rating'})

# Favorite genre per user (most frequent genre)
fav_genre = train_data.groupby(['user_id', 'genre']).size().reset_index(name='count')
fav_genre = fav_genre.sort_values(['user_id', 'count'], ascending=[True, False])
user_fav_genre = fav_genre.drop_duplicates(subset=['user_id']).set_index('user_id')['genre']

user_fav_genre = user_fav_genre.rename('user_fav_genre').reset_index()

# Most watched time of day per user
fav_watch_time = train_data.groupby(['user_id', 'watch_time']).size().reset_index(name='count')
fav_watch_time = fav_watch_time.sort_values(['user_id', 'count'], ascending=[True, False])
user_fav_watch_time = fav_watch_time.drop_duplicates(subset=['user_id']).set_index('user_id')['watch_time']
user_fav_watch_time = user_fav_watch_time.rename('fav_watch_time').reset_index()

# Movie-level features
movie_popularity = train_data.groupby('movie_id').size().reset_index(name='popularity_score')
movie_avg_rating = train_data.groupby('movie_id')['user_rating'].mean().reset_index().rename(
    columns={'user_rating': 'movie_avg_rating'})


# Merge user-level features
def merge_user_feats(df):
    df = df.merge(user_avg_rating, on='user_id', how='left')
    df = df.merge(user_fav_genre, on='user_id', how='left')
    df = df.merge(user_fav_watch_time, on='user_id', how='left')

    # One-hot encode user_fav_genre with fixed categories
    all_genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Sci-Fi']
    dummies_genre = pd.get_dummies(df['user_fav_genre'], prefix='fav_genre')
    for genre in all_genres:
        col = f'fav_genre_{genre}'
        if col not in dummies_genre.columns:
            dummies_genre[col] = 0
    dummies_genre = dummies_genre[[f'fav_genre_{g}' for g in all_genres]]

    # One-hot encode fav_watch_time with fixed categories
    all_times = ['morning', 'afternoon', 'evening', 'night']
    dummies_watch = pd.get_dummies(df['fav_watch_time'], prefix='fav_watch')
    for time in all_times:
        col = f'fav_watch_{time}'
        if col not in dummies_watch.columns:
            dummies_watch[col] = 0
    dummies_watch = dummies_watch[[f'fav_watch_{t}' for t in all_times]]

    df = pd.concat([df, dummies_genre, dummies_watch], axis=1)
    df.drop(columns=['user_fav_genre', 'fav_watch_time'], inplace=True)

    return df


# Merge movie-level features (with scaler fixed)
# Fit scaler on train movie features
movie_feats_train = movie_popularity.merge(movie_avg_rating, on='movie_id')
scaler_movie_feats = StandardScaler()
scaler_movie_feats.fit(movie_feats_train[['popularity_score', 'movie_avg_rating']])


def merge_movie_feats(df, scaler):
    df = df.merge(movie_popularity, on='movie_id', how='left')
    df = df.merge(movie_avg_rating, on='movie_id', how='left')

    # Fill missing movie features
    df['popularity_score'].fillna(0, inplace=True)
    df['movie_avg_rating'].fillna(movie_avg_rating['movie_avg_rating'].mean(), inplace=True)

    df[['popularity_score', 'movie_avg_rating']] = scaler.transform(df[['popularity_score', 'movie_avg_rating']])

    return df


# Add interaction feature user_rating × movie_avg_rating
def add_interaction_feature(df):
    df['user_movie_rating_interaction'] = df['user_rating'] * df['movie_avg_rating']
    return df


# === Apply merges and features to all datasets ===

for df in [train_data, val_data, test_data]:
    # Merge user features
    df = merge_user_feats(df)
    # Merge movie features
    df = merge_movie_feats(df, scaler_movie_feats)
    # Add interaction feature
    df = add_interaction_feature(df)

    # Save back to variables
    if df.equals(train_data):
        train_data = df.copy()
    elif df.equals(val_data):
        val_data = df.copy()
    else:
        test_data = df.copy()

# Fix: Above `equals` comparison won't work as intended — instead assign explicitly:

train_data = merge_user_feats(train_data)
train_data = merge_movie_feats(train_data, scaler_movie_feats)
train_data = add_interaction_feature(train_data)

val_data = merge_user_feats(val_data)
val_data = merge_movie_feats(val_data, scaler_movie_feats)
val_data = add_interaction_feature(val_data)

test_data = merge_user_feats(test_data)
test_data = merge_movie_feats(test_data, scaler_movie_feats)
test_data = add_interaction_feature(test_data)

# === 5. Prepare Final Feature Set ===

# Drop columns not needed for model input
drop_cols = ['user_id', 'movie_id', 'genre', 'watch_time', 'liked']  # 'liked' is target, keep separately
feature_cols = [c for c in train_data.columns if c not in drop_cols]

# Check for any missing values in features and fill (should be none, but safe)
train_data[feature_cols] = train_data[feature_cols].fillna(0)
val_data[feature_cols] = val_data[feature_cols].fillna(0)
test_data[feature_cols] = test_data[feature_cols].fillna(0)

X_train = train_data[feature_cols]
y_train = train_data['liked']

X_val = val_data[feature_cols]
y_val = val_data['liked']

X_test = test_data[feature_cols]
y_test = test_data['liked']

# === 6. Feature Selection (Correlation) ===
corr = train_data[feature_cols + ['liked']].corr()['liked'].abs().sort_values(ascending=False)
print("Top correlations with target:")
print(corr.head(20))

# Select features with correlation > threshold (e.g. 0.01)
selected_features = corr[corr > 0.01].index.drop('liked').tolist()
print("Selected features based on correlation threshold 0.01:")
print(selected_features)

# Use these features only
X_train_fs = X_train[selected_features]
X_val_fs = X_val[selected_features]
X_test_fs = X_test[selected_features]

# === 7. Model Training ===

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

for name, model in models.items():
    model.fit(X_train_fs, y_train)
    y_pred = model.predict(X_val_fs)
    y_prob = model.predict_proba(X_val_fs)[:, 1]
    print(f"\n{name} Validation Performance:")
    print(classification_report(y_val, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_val, y_prob):.4f}")

# === 8. Final Testing on Test Set with best model (say XGBoost) ===

best_model = models["XGBoost"]
y_pred_test = best_model.predict(X_test_fs)
y_prob_test = best_model.predict_proba(X_test_fs)[:, 1]

print("\nTest Set Performance (XGBoost):")
print(classification_report(y_test, y_pred_test))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_test):.4f}")

# === 9. Feature Importance Plot ===

importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances (XGBoost)")
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), [selected_features[i] for i in indices], rotation=45, ha='right')
plt.tight_layout()
plt.show()
