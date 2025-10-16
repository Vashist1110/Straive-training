"""
Netflix Movie Preference Prediction - Complete ML Pipeline
============================================================

This module implements an end-to-end machine learning workflow for predicting
user movie preferences on Netflix platform.

Author: ML Engineering Team
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    classification_report
import xgboost as xgb

# Visualization
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class NetflixMLPipeline:
    """
    Complete Machine Learning Pipeline for Netflix Movie Preference Prediction
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.users_df = None
        self.movies_df = None
        self.ratings_df = None
        self.feature_df = None
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}

        np.random.seed(random_state)

    # =========================================================================
    # STEP 1: DATA PREPARATION
    # =========================================================================

    def generate_sample_data(self):
        """
        Generate comprehensive sample Netflix dataset with realistic patterns
        """
        print("🎬 Generating Sample Netflix Dataset...")

        # User Demographics and Preferences
        self.users_df = pd.DataFrame({
            'user_id': range(1, 5001),
            'age': np.random.randint(18, 70, 5000),
            'gender': np.random.choice(['M', 'F', 'Other'], 5000, p=[0.48, 0.48, 0.04]),
            'subscription_type': np.random.choice(['Basic', 'Standard', 'Premium'], 5000, p=[0.3, 0.5, 0.2]),
            'country': np.random.choice(['US', 'UK', 'CA', 'AU', 'DE', 'FR', 'BR', 'IN'], 5000,
                                        p=[0.4, 0.15, 0.1, 0.08, 0.08, 0.07, 0.07, 0.05]),
            'avg_daily_hours': np.random.gamma(2, 1.2, 5000).clip(0.1, 8),
            'preferred_time': np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night'], 5000,
                                               p=[0.1, 0.2, 0.5, 0.2]),
            'signup_date': pd.date_range('2020-01-01', '2024-01-01', periods=5000),
            'account_status': np.random.choice(['Active', 'Paused'], 5000, p=[0.92, 0.08])
        })

        # Movie/Series Metadata
        genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Thriller', 'Documentary']
        self.movies_df = pd.DataFrame({
            'movie_id': range(1, 1001),
            'title': [f'Movie_{i}' for i in range(1, 1001)],
            'genre': np.random.choice(genres, 1000),
            'release_year': np.random.randint(1990, 2024, 1000),
            'duration_minutes': np.random.normal(110, 25, 1000).clip(80, 200).astype(int),
            'imdb_rating': np.random.normal(6.8, 1.2, 1000).clip(3.0, 9.5).round(1),
            'cast_popularity': np.random.beta(2, 5, 1000),
            'production_budget': np.random.lognormal(16, 1.5, 1000),
            'director': [f'Director_{i}' for i in np.random.randint(1, 201, 1000)],
            'language': np.random.choice(['English', 'Spanish', 'French', 'German'], 1000, p=[0.7, 0.15, 0.1, 0.05]),
            'content_rating': np.random.choice(['G', 'PG', 'PG-13', 'R', 'NC-17'], 1000, p=[0.1, 0.25, 0.4, 0.22, 0.03])
        })

        # Generate User-Movie Interactions with realistic patterns
        interactions = []
        for _ in range(50000):
            user_id = np.random.randint(1, 5001)
            movie_id = np.random.randint(1, 1001)

            # Get user and movie info for realistic rating generation
            user_age = self.users_df[self.users_df['user_id'] == user_id]['age'].iloc[0]
            movie_genre = self.movies_df[self.movies_df['movie_id'] == movie_id]['genre'].iloc[0]
            imdb_rating = self.movies_df[self.movies_df['movie_id'] == movie_id]['imdb_rating'].iloc[0]

            # Introduce realistic patterns
            like_prob = 0.5  # Base probability

            # Age-genre preferences
            if user_age < 25 and movie_genre in ['Action', 'Horror', 'Sci-Fi']:
                like_prob += 0.15
            elif user_age > 45 and movie_genre in ['Drama', 'Documentary']:
                like_prob += 0.12

            # IMDB rating influence
            like_prob += (imdb_rating - 6.5) * 0.08

            # Rating timestamp
            watch_date = pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(1, 365))

            liked = np.random.choice([0, 1], p=[1 - like_prob, like_prob])

            interactions.append([user_id, movie_id, liked, watch_date])

        self.ratings_df = pd.DataFrame(interactions, columns=['user_id', 'movie_id', 'liked', 'watch_date'])

        # Remove duplicates (same user rating same movie multiple times)
        self.ratings_df = self.ratings_df.drop_duplicates(subset=['user_id', 'movie_id'], keep='last')

        print(f"✅ Dataset Generated:")
        print(f"   👥 Users: {len(self.users_df):,}")
        print(f"   🎬 Movies: {len(self.movies_df):,}")
        print(f"   ⭐ Interactions: {len(self.ratings_df):,}")
        print(f"   👍 Like Ratio: {self.ratings_df['liked'].mean():.1%}")

        return self.users_df, self.movies_df, self.ratings_df

    def handle_missing_values(self):
        """
        Comprehensive missing value analysis and imputation
        """
        print("\n🔍 Handling Missing Values...")

        # Introduce some realistic missing values for demonstration
        self.users_df.loc[np.random.choice(self.users_df.index, 100), 'preferred_time'] = np.nan
        self.movies_df.loc[np.random.choice(self.movies_df.index, 50), 'imdb_rating'] = np.nan

        missing_info = {}

        for df_name, df in [('Users', self.users_df), ('Movies', self.movies_df), ('Ratings', self.ratings_df)]:
            missing_counts = df.isnull().sum()
            missing_pct = (missing_counts / len(df)) * 100
            missing_info[df_name] = missing_pct[missing_pct > 0]

            if len(missing_info[df_name]) > 0:
                print(f"   {df_name} Missing Values:")
                for col, pct in missing_info[df_name].items():
                    print(f"     - {col}: {pct:.1f}%")

        # Handle missing values
        # Categorical: Mode imputation
        categorical_cols = ['preferred_time', 'gender', 'country']
        for col in categorical_cols:
            if col in self.users_df.columns:
                mode_val = self.users_df[col].mode()
                if len(mode_val) > 0:
                    self.users_df[col].fillna(mode_val[0], inplace=True)

        # Numerical: Median imputation
        numerical_cols = ['imdb_rating', 'cast_popularity', 'production_budget']
        for col in numerical_cols:
            if col in self.movies_df.columns:
                median_val = self.movies_df[col].median()
                self.movies_df[col].fillna(median_val, inplace=True)

        print("✅ Missing values handled")
        return missing_info

    def encode_categorical_features(self):
        """
        Advanced categorical encoding with multiple strategies
        """
        print("\n🏷️ Encoding Categorical Features...")

        # Label Encoding for ordinal features
        ordinal_features = {
            'subscription_type': {'Basic': 1, 'Standard': 2, 'Premium': 3},
            'content_rating': {'G': 1, 'PG': 2, 'PG-13': 3, 'R': 4, 'NC-17': 5}
        }

        for feature, mapping in ordinal_features.items():
            if feature in self.users_df.columns:
                self.users_df[f'{feature}_encoded'] = self.users_df[feature].map(mapping)
            elif feature in self.movies_df.columns:
                self.movies_df[f'{feature}_encoded'] = self.movies_df[feature].map(mapping)

        # Label Encoding for nominal features
        nominal_features = ['gender', 'country', 'preferred_time', 'genre', 'language', 'director']

        for feature in nominal_features:
            if feature in self.users_df.columns:
                le = LabelEncoder()
                self.users_df[f'{feature}_encoded'] = le.fit_transform(self.users_df[feature])
                self.encoders[f'users_{feature}'] = le
            elif feature in self.movies_df.columns:
                le = LabelEncoder()
                self.movies_df[f'{feature}_encoded'] = le.fit_transform(self.movies_df[feature])
                self.encoders[f'movies_{feature}'] = le

        print("✅ Categorical encoding complete")
        return self.encoders

    def normalize_numerical_features(self):
        """
        Normalize numerical features using StandardScaler
        """
        print("\n📊 Normalizing Numerical Features...")

        # User numerical features
        user_numerical = ['age', 'avg_daily_hours']
        user_scaler = StandardScaler()
        self.users_df[user_numerical] = user_scaler.fit_transform(self.users_df[user_numerical])
        self.scalers['users'] = user_scaler

        # Movie numerical features
        movie_numerical = ['release_year', 'duration_minutes', 'imdb_rating', 'cast_popularity', 'production_budget']
        movie_scaler = StandardScaler()
        self.movies_df[movie_numerical] = movie_scaler.fit_transform(self.movies_df[movie_numerical])
        self.scalers['movies'] = movie_scaler

        print("✅ Numerical features normalized")
        return self.scalers

    # =========================================================================
    # STEP 2: DATA VALIDATION
    # =========================================================================

    def create_smart_train_test_split(self, test_size=0.2, val_size=0.1):
        """
        Advanced train-test split avoiding cold-start problem
        """
        print("\n✅ Creating Smart Train-Test Split...")

        # Ensure users have sufficient interactions (minimum 3 ratings)
        user_counts = self.ratings_df['user_id'].value_counts()
        active_users = user_counts[user_counts >= 3].index

        print(f"   📊 Active users (≥3 ratings): {len(active_users):,} ({len(active_users) / len(user_counts):.1%})")

        # Filter data for active users only
        filtered_ratings = self.ratings_df[self.ratings_df['user_id'].isin(active_users)].copy()

        # Stratified sampling to maintain class balance
        train_data, temp_data = train_test_split(
            filtered_ratings,
            test_size=(test_size + val_size),
            stratify=filtered_ratings['liked'],
            random_state=self.random_state
        )

        val_data, test_data = train_test_split(
            temp_data,
            test_size=test_size / (test_size + val_size),
            stratify=temp_data['liked'],
            random_state=self.random_state
        )

        # Validation checks
        train_users = set(train_data['user_id'].unique())
        val_users = set(val_data['user_id'].unique())
        test_users = set(test_data['user_id'].unique())

        train_test_overlap = len(train_users & test_users) / len(test_users)
        train_val_overlap = len(train_users & val_users) / len(val_users)

        print(f"   📈 Data splits created:")
        print(f"     - Training: {len(train_data):,} samples")
        print(f"     - Validation: {len(val_data):,} samples")
        print(f"     - Test: {len(test_data):,} samples")
        print(f"     - User overlap (train-test): {train_test_overlap:.1%}")
        print(f"     - User overlap (train-val): {train_val_overlap:.1%}")

        # Class balance check
        for split_name, split_data in [('Train', train_data), ('Val', val_data), ('Test', test_data)]:
            like_ratio = split_data['liked'].mean()
            print(f"     - {split_name} like ratio: {like_ratio:.1%}")

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        return train_data, val_data, test_data

    # =========================================================================
    # STEP 3: FEATURE SELECTION
    # =========================================================================

    def analyze_feature_correlations(self, target_col='liked'):
        """
        Comprehensive correlation analysis and multicollinearity detection
        """
        print("\n🎯 Analyzing Feature Correlations...")

        # Create feature matrix
        feature_df = self.create_base_feature_matrix()

        # Calculate correlation matrix
        correlation_matrix = feature_df.corr()

        # Find highly correlated features (multicollinearity)
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr = correlation_matrix.iloc[i, j]
                if abs(corr) > 0.8:
                    high_corr_pairs.append((
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        corr
                    ))

        if high_corr_pairs:
            print(f"   ⚠️ Found {len(high_corr_pairs)} highly correlated feature pairs (|r| > 0.8):")
            for feat1, feat2, corr in high_corr_pairs:
                print(f"     - {feat1} ↔ {feat2}: {corr:.3f}")
        else:
            print("   ✅ No concerning multicollinearity detected")

        # Feature-target correlations
        target_corrs = correlation_matrix[target_col].abs().sort_values(ascending=False)
        print(f"\n   🎯 Top features correlated with {target_col}:")
        for feature, corr in target_corrs.head(10).items():
            if feature != target_col:
                print(f"     - {feature}: {corr:.3f}")

        self.correlation_matrix = correlation_matrix
        self.high_corr_pairs = high_corr_pairs

        return correlation_matrix, high_corr_pairs

    def calculate_feature_importance(self, method='all'):
        """
        Multiple feature importance calculation methods
        """
        print(f"\n🔍 Calculating Feature Importance ({method})...")

        # Prepare feature matrix
        X, y = self.prepare_ml_data()

        importance_results = {}

        if method in ['rf', 'all']:
            # Random Forest Feature Importance
            rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            rf.fit(X, y)
            rf_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
            importance_results['random_forest'] = rf_importance
            print(f"   🌲 Random Forest - Top 5 features:")
            for feat, imp in rf_importance.head(5).items():
                print(f"     - {feat}: {imp:.3f}")

        if method in ['xgb', 'all']:
            # XGBoost Feature Importance
            xgb_model = xgb.XGBClassifier(random_state=self.random_state, eval_metric='logloss')
            xgb_model.fit(X, y)
            xgb_importance = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(ascending=False)
            importance_results['xgboost'] = xgb_importance
            print(f"   🚀 XGBoost - Top 5 features:")
            for feat, imp in xgb_importance.head(5).items():
                print(f"     - {feat}: {imp:.3f}")

        if method in ['statistical', 'all']:
            # Statistical F-score
            selector = SelectKBest(score_func=f_classif, k='all')
            selector.fit(X, y)
            f_scores = pd.Series(selector.scores_, index=X.columns).sort_values(ascending=False)
            importance_results['f_score'] = f_scores
            print(f"   📊 F-Score - Top 5 features:")
            for feat, score in f_scores.head(5).items():
                print(f"     - {feat}: {score:.1f}")

        self.feature_importance = importance_results
        return importance_results

    def perform_pca_analysis(self, n_components=0.95):
        """
        Principal Component Analysis for dimensionality reduction
        """
        print(f"\n🔢 Performing PCA Analysis (explaining {n_components:.0%} variance)...")

        X, y = self.prepare_ml_data()

        # Apply PCA
        pca = PCA(n_components=n_components, random_state=self.random_state)
        X_pca = pca.fit_transform(X)

        print(f"   📉 Dimensionality reduction: {X.shape[1]} → {X_pca.shape[1]} features")
        print(f"   📈 Explained variance: {pca.explained_variance_ratio_.sum():.1%}")

        # Top components analysis
        feature_names = X.columns
        top_features_per_component = []

        for i, component in enumerate(pca.components_[:5]):  # Top 5 components
            top_features_idx = np.argsort(np.abs(component))[-3:][::-1]  # Top 3 features
            top_features = [(feature_names[idx], component[idx]) for idx in top_features_idx]
            top_features_per_component.append(top_features)

            print(f"   🎯 PC{i + 1} (explains {pca.explained_variance_ratio_[i]:.1%} variance):")
            for feat, weight in top_features:
                print(f"     - {feat}: {weight:.3f}")

        self.pca_model = pca
        self.pca_features = top_features_per_component

        return pca, X_pca

    # =========================================================================
    # STEP 4: FEATURE ENGINEERING
    # =========================================================================

    def create_user_level_features(self):
        """
        Advanced user-level feature engineering
        """
        print("\n🛠️ Creating User-Level Features...")

        # Basic user statistics
        user_stats = self.ratings_df.groupby('user_id').agg({
            'liked': ['mean', 'count', 'std'],
            'movie_id': 'nunique',
            'watch_date': ['min', 'max']
        }).round(3)

        user_stats.columns = ['avg_rating', 'total_ratings', 'rating_std', 'unique_movies', 'first_watch', 'last_watch']
        user_stats['rating_std'] = user_stats['rating_std'].fillna(0)  # Handle single rating users

        # Viewing patterns
        user_stats['days_active'] = (user_stats['last_watch'] - user_stats['first_watch']).dt.days + 1
        user_stats['avg_movies_per_day'] = user_stats['total_ratings'] / user_stats['days_active']

        # User preferences
        user_movie_genre = self.ratings_df.merge(
            self.movies_df[['movie_id', 'genre', 'release_year', 'imdb_rating']],
            on='movie_id'
        )

        # Favorite genre (most liked genre)
        user_genre_likes = user_movie_genre[user_movie_genre['liked'] == 1].groupby(
            ['user_id', 'genre']).size().reset_index(name='likes')
        user_fav_genre = \
        user_genre_likes.loc[user_genre_likes.groupby('user_id')['likes'].idxmax()].set_index('user_id')['genre']

        # Time-based preferences
        user_movie_genre['hour'] = pd.to_datetime(user_movie_genre['watch_date']).dt.hour
        user_movie_genre['is_weekend'] = pd.to_datetime(user_movie_genre['watch_date']).dt.dayofweek >= 5

        user_time_pref = user_movie_genre.groupby('user_id').agg({
            'hour': 'mean',
            'is_weekend': 'mean',
            'release_year': 'mean',
            'imdb_rating': 'mean'
        }).round(3)
        user_time_pref.columns = ['avg_watch_hour', 'weekend_ratio', 'avg_movie_year', 'avg_movie_rating']

        # Quality preference (likes high-rated movies?)
        user_stats['quality_preference'] = user_time_pref['avg_movie_rating']
        user_stats['recency_preference'] = user_time_pref['avg_movie_year']
        user_stats['night_owl_score'] = (user_time_pref['avg_watch_hour'] > 20).astype(int)
        user_stats['weekend_watcher'] = (user_time_pref['weekend_ratio'] > 0.6).astype(int)

        # Merge favorite genre
        user_stats = user_stats.merge(user_fav_genre.rename('favorite_genre'), left_index=True, right_index=True,
                                      how='left')
        user_stats['favorite_genre'] = user_stats['favorite_genre'].fillna('Unknown')

        # Engagement score
        user_stats['engagement_score'] = (
                user_stats['total_ratings'].rank(pct=True) * 0.4 +
                user_stats['unique_movies'].rank(pct=True) * 0.3 +
                user_stats['avg_movies_per_day'].rank(pct=True) * 0.3
        )

        print(f"   ✅ Created {len(user_stats.columns)} user-level features")
        self.user_features = user_stats
        return user_stats

    def create_movie_level_features(self):
        """
        Advanced movie-level feature engineering
        """
        print("\n🎬 Creating Movie-Level Features...")

        # Basic movie statistics
        movie_stats = self.ratings_df.groupby('movie_id').agg({
            'liked': ['mean', 'count', 'std'],
            'user_id': 'nunique',
            'watch_date': ['min', 'max', 'count']
        }).round(3)

        movie_stats.columns = ['popularity_score', 'total_views', 'rating_std', 'unique_viewers', 'first_view',
                               'last_view', 'view_count']
        movie_stats['rating_std'] = movie_stats['rating_std'].fillna(0)

        # Engagement metrics
        movie_stats['user_penetration'] = movie_stats['unique_viewers'] / len(self.users_df)
        movie_stats['repeat_view_ratio'] = (movie_stats['total_views'] - movie_stats['unique_viewers']) / movie_stats[
            'total_views']
        movie_stats['repeat_view_ratio'] = movie_stats['repeat_view_ratio'].fillna(0)

        # Trending analysis
        recent_cutoff = pd.Timestamp.now() - pd.Timedelta(days=30)
        recent_views = self.ratings_df[self.ratings_df['watch_date'] > recent_cutoff].groupby('movie_id').size()
        movie_stats['recent_views'] = recent_views.fillna(0)
        movie_stats['trending_score'] = movie_stats['recent_views'] / (movie_stats['total_views'] + 1)

        # Viral coefficient (how much it spreads)
        movie_stats['days_since_first_view'] = (pd.Timestamp.now() - movie_stats['first_view']).dt.days + 1
        movie_stats['viral_coefficient'] = movie_stats['unique_viewers'] / movie_stats['days_since_first_view']

        # Quality indicators
        movie_stats['polarization'] = movie_stats['rating_std']  # High std = polarizing content
        movie_stats['hit_potential'] = (movie_stats['popularity_score'] > 0.8) & (movie_stats['total_views'] > 100)

        # Categorical features
        movie_stats['is_trending'] = movie_stats['trending_score'] > movie_stats['trending_score'].quantile(0.8)
        movie_stats['is_popular'] = movie_stats['popularity_score'] > movie_stats['popularity_score'].quantile(0.7)
        movie_stats['is_niche'] = movie_stats['unique_viewers'] < movie_stats['unique_viewers'].quantile(0.3)

        print(f"   ✅ Created {len(movie_stats.columns)} movie-level features")
        self.movie_features = movie_stats
        return movie_stats

    def create_interaction_features(self):
        """
        Advanced user-movie interaction features
        """
        print("\n🤝 Creating Interaction Features...")

        # Merge all data
        interaction_data = self.ratings_df.merge(
            self.users_df, on='user_id', how='left'
        ).merge(
            self.movies_df, on='movie_id', how='left'
        )

        # User-Genre Interaction
        interaction_data = interaction_data.merge(
            self.user_features[['favorite_genre']], left_on='user_id', right_index=True, how='left'
        )
        interaction_data['genre_match'] = (interaction_data['favorite_genre'] == interaction_data['genre']).astype(int)

        # Age-Genre Interaction
        interaction_data['age_genre_match'] = 0
        interaction_data.loc[(interaction_data['age'] < 30) & (
            interaction_data['genre'].isin(['Action', 'Horror', 'Sci-Fi'])), 'age_genre_match'] = 1
        interaction_data.loc[(interaction_data['age'] >= 45) & (
            interaction_data['genre'].isin(['Drama', 'Documentary'])), 'age_genre_match'] = 1

        # Subscription-Content Match
        interaction_data['premium_content'] = (
                    interaction_data['production_budget'] > interaction_data['production_budget'].quantile(0.7)).astype(
            int)
        interaction_data['subscription_content_match'] = (
                (interaction_data['subscription_type'] == 'Premium') & (interaction_data['premium_content'] == 1)
        ).astype(int)

        # Time-based interactions
        interaction_data['watch_hour'] = pd.to_datetime(interaction_data['watch_date']).dt.hour
        interaction_data['time_preference_match'] = 0

        # Match watching time with preferred time
        time_mapping = {'Morning': (6, 12), 'Afternoon': (12, 18), 'Evening': (18, 22), 'Night': (22, 6)}
        for pref_time, (start, end) in time_mapping.items():
            if start < end:
                mask = (interaction_data['preferred_time'] == pref_time) & (
                    interaction_data['watch_hour'].between(start, end - 1))
            else:  # Night (22-6)
                mask = (interaction_data['preferred_time'] == pref_time) & (
                            (interaction_data['watch_hour'] >= start) | (interaction_data['watch_hour'] < end))
            interaction_data.loc[mask, 'time_preference_match'] = 1

        # Recency bias (newer movies for young users)
        current_year = pd.Timestamp.now().year
        interaction_data['movie_age'] = current_year - interaction_data['release_year']
        interaction_data['recency_match'] = (
                    (interaction_data['age'] < 35) & (interaction_data['movie_age'] < 5)).astype(int)

        # Quality expectation match
        interaction_data = interaction_data.merge(
            self.user_features[['quality_preference']], left_on='user_id', right_index=True, how='left'
        )
        interaction_data['quality_match'] = (
                    interaction_data['imdb_rating'] >= interaction_data['quality_preference']).astype(int)

        print(f"   ✅ Created interaction features")
        self.interaction_data = interaction_data
        return interaction_data

    def create_base_feature_matrix(self):
        """
        Combine all features into a single matrix for ML
        """
        print("\n🏗️ Building Complete Feature Matrix...")

        # Start with interaction data
        feature_df = self.interaction_data.copy()

        # Add user features
        user_feat_cols = ['avg_rating', 'total_ratings', 'rating_std', 'unique_movies',
                          'engagement_score', 'quality_preference', 'night_owl_score', 'weekend_watcher']
        feature_df = feature_df.merge(
            self.user_features[user_feat_cols], left_on='user_id', right_index=True, how='left'
        )

        # Add movie features
        movie_feat_cols = ['popularity_score', 'total_views', 'unique_viewers', 'trending_score',
                           'viral_coefficient', 'polarization', 'is_trending', 'is_popular']
        feature_df = feature_df.merge(
            self.movie_features[movie_feat_cols], left_on='movie_id', right_index=True, how='left'
        )

        # Select final feature columns
        feature_columns = [
            # User features
            'age', 'avg_daily_hours', 'gender_encoded', 'subscription_type_encoded',
            'avg_rating', 'total_ratings', 'rating_std', 'engagement_score',
            'quality_preference', 'night_owl_score', 'weekend_watcher',

            # Movie features
            'release_year', 'duration_minutes', 'imdb_rating', 'cast_popularity',
            'genre_encoded', 'content_rating_encoded',
            'popularity_score', 'total_views', 'trending_score', 'viral_coefficient',

            # Interaction features
            'genre_match', 'age_genre_match', 'subscription_content_match',
            'time_preference_match', 'recency_match', 'quality_match',

            # Target
            'liked'
        ]

        # Filter and clean
        feature_df = feature_df[feature_columns].copy()
        feature_df = feature_df.fillna(0)  # Handle any remaining NaN values

        print(f"   ✅ Feature matrix created: {feature_df.shape}")
        print(f"   📊 Features: {len(feature_columns) - 1} + target")

        self.feature_df = feature_df
        return feature_df

    def prepare_ml_data(self, include_target=True):
        """
        Prepare final X, y matrices for machine learning
        """
        if self.feature_df is None:
            self.create_complete_feature_pipeline()

        if include_target:
            X = self.feature_df.drop('liked', axis=1)
            y = self.feature_df['liked']
            return X, y
        else:
            return self.feature_df.drop('liked', axis=1)

    def create_complete_feature_pipeline(self):
        """
        Execute the complete feature engineering pipeline
        """
        print("\n🏭 Executing Complete Feature Engineering Pipeline...")

        # Step 1: Create user-level features
        self.create_user_level_features()

        # Step 2: Create movie-level features
        self.create_movie_level_features()

        # Step 3: Create interaction features
        self.create_interaction_features()

        # Step 4: Build final feature matrix
        self.create_base_feature_matrix()

        print("✅ Feature engineering pipeline completed!")
        print(f"   Final dataset shape: {self.feature_df.shape}")
        return self.feature_df

    # =========================================================================
    # STEP 5: MODEL BUILDING
    # =========================================================================

    def train_logistic_regression(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train and optimize Logistic Regression model
        """
        print("\n📊 Training Logistic Regression...")

        # Hyperparameter grid
        lr_params = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear'],
            'max_iter': [1000]
        }

        # Grid search with cross-validation
        lr_grid = GridSearchCV(
            LogisticRegression(random_state=self.random_state),
            lr_params,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            scoring='roc_auc',
            n_jobs=-1,
            verbose=0
        )

        lr_grid.fit(X_train, y_train)

        print(f"   🎯 Best parameters: {lr_grid.best_params_}")
        print(f"   📈 Best CV score: {lr_grid.best_score_:.4f}")

        # Validation performance
        if X_val is not None and y_val is not None:
            val_score = lr_grid.score(X_val, y_val)
            print(f"   ✅ Validation AUC: {val_score:.4f}")

        self.models['logistic_regression'] = lr_grid.best_estimator_
        return lr_grid.best_estimator_

    def train_random_forest(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train and optimize Random Forest model
        """
        print("\n🌲 Training Random Forest...")

        # Hyperparameter grid
        rf_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True]
        }

        # Grid search with cross-validation
        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
            rf_params,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            scoring='roc_auc',
            n_jobs=-1,
            verbose=0
        )

        rf_grid.fit(X_train, y_train)

        print(f"   🎯 Best parameters: {rf_grid.best_params_}")
        print(f"   📈 Best CV score: {rf_grid.best_score_:.4f}")

        # Feature importance
        feature_importance = pd.Series(
            rf_grid.best_estimator_.feature_importances_,
            index=X_train.columns
        ).sort_values(ascending=False)

        print(f"   🔝 Top 5 important features:")
        for feat, imp in feature_importance.head(5).items():
            print(f"     - {feat}: {imp:.4f}")

        # Validation performance
        if X_val is not None and y_val is not None:
            val_score = rf_grid.score(X_val, y_val)
            print(f"   ✅ Validation AUC: {val_score:.4f}")

        self.models['random_forest'] = rf_grid.best_estimator_
        return rf_grid.best_estimator_

    def train_xgboost(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train and optimize XGBoost model
        """
        print("\n🚀 Training XGBoost...")

        # Hyperparameter grid
        xgb_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }

        # Grid search with cross-validation
        xgb_grid = GridSearchCV(
            xgb.XGBClassifier(random_state=self.random_state, eval_metric='logloss'),
            xgb_params,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            scoring='roc_auc',
            n_jobs=-1,
            verbose=0
        )

        xgb_grid.fit(X_train, y_train)

        print(f"   🎯 Best parameters: {xgb_grid.best_params_}")
        print(f"   📈 Best CV score: {xgb_grid.best_score_:.4f}")

        # Feature importance
        feature_importance = pd.Series(
            xgb_grid.best_estimator_.feature_importances_,
            index=X_train.columns
        ).sort_values(ascending=False)

        print(f"   🔝 Top 5 important features:")
        for feat, imp in feature_importance.head(5).items():
            print(f"     - {feat}: {imp:.4f}")

        # Validation performance
        if X_val is not None and y_val is not None:
            val_score = xgb_grid.score(X_val, y_val)
            print(f"   ✅ Validation AUC: {val_score:.4f}")

        self.models['xgboost'] = xgb_grid.best_estimator_
        return xgb_grid.best_estimator_

    def train_all_models(self):
        """
        Train all models with proper data preparation
        """
        print("\n🧠 Training All Models...")

        # Prepare data
        X, y = self.prepare_ml_data()

        # Create train/validation split from training data
        train_idx = self.train_data.index
        val_idx = self.val_data.index

        X_train = X.loc[train_idx]
        y_train = y.loc[train_idx]
        X_val = X.loc[val_idx]
        y_val = y.loc[val_idx]

        print(f"   📊 Training set: {X_train.shape}")
        print(f"   📊 Validation set: {X_val.shape}")

        # Train models
        models_trained = {}

        # 1. Logistic Regression
        models_trained['lr'] = self.train_logistic_regression(X_train, y_train, X_val, y_val)

        # 2. Random Forest
        models_trained['rf'] = self.train_random_forest(X_train, y_train, X_val, y_val)

        # 3. XGBoost
        models_trained['xgb'] = self.train_xgboost(X_train, y_train, X_val, y_val)

        print("\n🎉 All models trained successfully!")
        return models_trained

    # =========================================================================
    # STEP 6: MODEL EVALUATION
    # =========================================================================

    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """
        Comprehensive model evaluation
        """
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }

        print(f"\n📈 {model_name} Performance:")
        print(f"   🎯 Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   🎯 Precision: {metrics['precision']:.4f}")
        print(f"   🎯 Recall:    {metrics['recall']:.4f}")
        print(f"   🎯 F1-Score:  {metrics['f1']:.4f}")
        print(f"   🎯 ROC-AUC:   {metrics['roc_auc']:.4f}")

        return metrics, y_pred, y_pred_proba

    def evaluate_all_models(self):
        """
        Evaluate all trained models on test set
        """
        print("\n📊 Evaluating All Models on Test Set...")

        # Prepare test data
        X, y = self.prepare_ml_data()
        test_idx = self.test_data.index
        X_test = X.loc[test_idx]
        y_test = y.loc[test_idx]

        print(f"   📊 Test set: {X_test.shape}")

        # Evaluate each model
        results = {}

        model_names = {
            'logistic_regression': 'Logistic Regression',
            'random_forest': 'Random Forest',
            'xgboost': 'XGBoost'
        }

        for model_key, model in self.models.items():
            model_name = model_names.get(model_key, model_key)
            metrics, y_pred, y_pred_proba = self.evaluate_model(model, X_test, y_test, model_name)

            results[model_key] = {
                'metrics': metrics,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }

        # Compare models
        print("\n🏆 Model Comparison Summary:")
        print("=" * 60)
        print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC':<10}")
        print("=" * 60)

        best_model = None
        best_auc = 0

        for model_key, result in results.items():
            metrics = result['metrics']
            model_name = model_names.get(model_key, model_key)

            print(f"{model_name:<20} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
                  f"{metrics['recall']:<10.4f} {metrics['f1']:<10.4f} {metrics['roc_auc']:<10.4f}")

            if metrics['roc_auc'] > best_auc:
                best_auc = metrics['roc_auc']
                best_model = model_key

        print("=" * 60)
        print(f"🥇 Best Model: {model_names.get(best_model, best_model)} (AUC: {best_auc:.4f})")

        self.evaluation_results = results
        self.best_model = best_model

        return results

    def analyze_model_errors(self):
        """
        Detailed error analysis and model insights
        """
        print("\n🔍 Analyzing Model Errors...")

        if not hasattr(self, 'evaluation_results'):
            print("   ❌ No evaluation results found. Run evaluate_all_models() first.")
            return

        X, y = self.prepare_ml_data()
        test_idx = self.test_data.index
        X_test = X.loc[test_idx]
        y_test = y.loc[test_idx]

        best_model_key = self.best_model
        best_result = self.evaluation_results[best_model_key]
        y_pred = best_result['predictions']

        # Confusion matrix analysis
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        print(f"\n📊 Confusion Matrix Analysis ({self.models[best_model_key].__class__.__name__}):")
        print(f"   True Negatives:  {tn:,}")
        print(f"   False Positives: {fp:,}")
        print(f"   False Negatives: {fn:,}")
        print(f"   True Positives:  {tp:,}")

        # Error rates
        false_positive_rate = fp / (fp + tn)
        false_negative_rate = fn / (fn + tp)

        print(f"\n📈 Error Analysis:")
        print(f"   False Positive Rate: {false_positive_rate:.1%} (recommending bad movies)")
        print(f"   False Negative Rate: {false_negative_rate:.1%} (missing good recommendations)")

        # Business impact analysis
        print(f"\n💼 Business Impact:")
        print(f"   • {fp:,} users might be shown movies they won't like")
        print(f"   • {fn:,} users might miss movies they would have liked")
        print(f"   • Precision focus: Reduces user frustration from bad recommendations")
        print(f"   • Recall focus: Maximizes user engagement and discovery")

        return cm, false_positive_rate, false_negative_rate

    # =========================================================================
    # STEP 7: MODEL INSIGHTS & RECOMMENDATIONS
    # =========================================================================

    def generate_feature_insights(self):
        """
        Generate insights from feature importance and model behavior
        """
        print("\n💡 Generating Feature Insights...")

        # Get feature importance from best model
        best_model = self.models[self.best_model]

        if hasattr(best_model, 'feature_importances_'):
            X, _ = self.prepare_ml_data()
            feature_importance = pd.Series(
                best_model.feature_importances_,
                index=X.columns
            ).sort_values(ascending=False)

            print("\n🔝 Top 10 Most Important Features:")
            for i, (feat, imp) in enumerate(feature_importance.head(10).items(), 1):
                print(f"   {i:2d}. {feat:<25} {imp:.4f}")

            # Feature categories analysis
            user_features = [f for f in feature_importance.index if any(x in f for x in
                                                                        ['age', 'avg_rating', 'total_ratings',
                                                                         'engagement', 'quality', 'night_owl',
                                                                         'weekend'])]
            movie_features = [f for f in feature_importance.index if any(
                x in f for x in ['release_year', 'duration', 'imdb_rating', 'popularity', 'trending', 'viral'])]
            interaction_features = [f for f in feature_importance.index if
                                    any(x in f for x in ['match', 'genre_encoded', 'subscription'])]

            print(f"\n📊 Feature Category Importance:")
            print(f"   User Features:        {feature_importance[user_features].sum():.3f}")
            print(f"   Movie Features:       {feature_importance[movie_features].sum():.3f}")
            print(f"   Interaction Features: {feature_importance[interaction_features].sum():.3f}")

            self.top_features = feature_importance
            return feature_importance

    def generate_business_recommendations(self):
        """
        Generate actionable business recommendations
        """
        print("\n🚀 Business Recommendations for Netflix:")
        print("=" * 50)

        # Model performance insights
        best_metrics = self.evaluation_results[self.best_model]['metrics']

        print("\n1. 🎯 MODEL DEPLOYMENT STRATEGY:")
        print(f"   • Deploy {self.models[self.best_model].__class__.__name__} as primary model")
        print(f"   • Expected accuracy: {best_metrics['accuracy']:.1%}")
        print(f"   • Expected precision: {best_metrics['precision']:.1%} (reduces bad recommendations)")
        print(f"   • Expected recall: {best_metrics['recall']:.1%} (captures user preferences)")

        print("\n2. 📊 A/B TESTING FRAMEWORK:")
        print("   • Start with 10% traffic to new model")
        print("   • Monitor click-through rates and user satisfaction")
        print("   • Compare with existing collaborative filtering")
        print("   • Gradually increase traffic if metrics improve")

        print("\n3. 🛠️ FEATURE ENGINEERING PRIORITIES:")
        if hasattr(self, 'top_features'):
            top_3_features = self.top_features.head(3).index.tolist()
            print("   • Focus on improving data quality for:")
            for feat in top_3_features:
                print(f"     - {feat}")

        print("\n4. 📈 PERFORMANCE MONITORING:")
        print("   • Set up real-time monitoring for model drift")
        print("   • Track daily precision/recall metrics")
        print("   • Monitor feature importance changes")
        print("   • Set alerts for AUC drops below 0.85")

        print("\n5. 🎬 CONTENT STRATEGY:")
        print("   • Invest in content matching high-importance features")
        print("   • Consider user engagement scores for acquisition decisions")
        print("   • Use viral coefficient to identify breakout content")

        print("\n6. 👥 USER EXPERIENCE:")
        print("   • Implement confidence scores for recommendations")
        print("   • Show 'Why recommended' explanations using top features")
        print("   • Create personalized genre discovery flows")

        print("\n7. 🔄 CONTINUOUS IMPROVEMENT:")
        print("   • Retrain models monthly with new data")
        print("   • Experiment with deep learning approaches (Neural CF)")
        print("   • Add real-time user behavior features")
        print("   • Consider multi-objective optimization (accuracy + diversity)")

        return self.generate_implementation_roadmap()

    def generate_implementation_roadmap(self):
        """
        Create a detailed implementation roadmap
        """
        print("\n📅 IMPLEMENTATION ROADMAP:")
        print("=" * 50)

        roadmap = {
            "Phase 1 (Weeks 1-4): Foundation": [
                "Set up model training pipeline",
                "Implement feature engineering automation",
                "Create model evaluation framework",
                "Build A/B testing infrastructure"
            ],

            "Phase 2 (Weeks 5-8): Deployment": [
                "Deploy models to staging environment",
                "Implement real-time inference API",
                "Set up monitoring and alerting",
                "Begin 10% traffic A/B test"
            ],

            "Phase 3 (Weeks 9-12): Optimization": [
                "Analyze A/B test results",
                "Optimize inference latency (<100ms)",
                "Implement recommendation explanations",
                "Scale to 50% traffic if successful"
            ],

            "Phase 4 (Weeks 13-16): Enhancement": [
                "Add collaborative filtering features",
                "Implement multi-objective optimization",
                "Create personalized genre flows",
                "Full traffic deployment"
            ]
        }

        for phase, tasks in roadmap.items():
            print(f"\n{phase}:")
            for task in tasks:
                print(f"   • {task}")

        return roadmap

    def save_model_artifacts(self, save_path="netflix_models/"):
        """
        Save all trained models and artifacts
        """
        import pickle
        import os

        print(f"\n💾 Saving Model Artifacts to {save_path}...")

        # Create directory
        os.makedirs(save_path, exist_ok=True)

        # Save models
        for model_name, model in self.models.items():
            model_file = f"{save_path}{model_name}_model.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            print(f"   ✅ Saved {model_name} model")

        # Save feature engineering pipeline components
        artifacts = {
            'user_features': self.user_features,
            'movie_features': self.movie_features,
            'feature_columns': self.feature_df.columns.tolist(),
            'encoders': self.encoders,
            'scalers': self.scalers,
            'evaluation_results': self.evaluation_results,
            'best_model': self.best_model
        }

        artifacts_file = f"{save_path}pipeline_artifacts.pkl"
        with open(artifacts_file, 'wb') as f:
            pickle.dump(artifacts, f)
        print(f"   ✅ Saved pipeline artifacts")

        # Save configuration
        config = {
            'random_state': self.random_state,
            'model_performance': {k: v['metrics'] for k, v in self.evaluation_results.items()},
            'feature_importance': self.top_features.to_dict() if hasattr(self, 'top_features') else {},
            'dataset_stats': {
                'users': len(self.users_df),
                'movies': len(self.movies_df),
                'interactions': len(self.ratings_df)
            }
        }

        config_file = f"{save_path}model_config.pkl"
        with open(config_file, 'wb') as f:
            pickle.dump(config, f)
        print(f"   ✅ Saved model configuration")

        print(f"   📁 All artifacts saved to: {save_path}")
        return save_path

    # =========================================================================
    # MAIN EXECUTION PIPELINE
    # =========================================================================

    def run_complete_pipeline(self):
        """
        Execute the complete end-to-end ML pipeline
        """
        print("🎬 Starting Complete Netflix ML Pipeline")
        print("=" * 60)

        try:
            # Step 1: Data Preparation
            print("\n" + "=" * 20 + " STEP 1: DATA PREPARATION " + "=" * 20)
            self.generate_sample_data()
            self.handle_missing_values()
            self.encode_categorical_features()
            self.normalize_numerical_features()

            # Step 2: Data Validation
            print("\n" + "=" * 20 + " STEP 2: DATA VALIDATION " + "=" * 21)
            self.create_smart_train_test_split()

            # Step 3: Feature Selection
            print("\n" + "=" * 20 + " STEP 3: FEATURE SELECTION " + "=" * 19)
            self.analyze_feature_correlations()
            self.calculate_feature_importance()
            self.perform_pca_analysis()

            # Step 4: Feature Engineering
            print("\n" + "=" * 19 + " STEP 4: FEATURE ENGINEERING " + "=" * 19)
            self.create_complete_feature_pipeline()

            # Step 5: Model Building
            print("\n" + "=" * 21 + " STEP 5: MODEL BUILDING " + "=" * 21)
            self.train_all_models()

            # Step 6: Model Evaluation
            print("\n" + "=" * 20 + " STEP 6: MODEL EVALUATION " + "=" * 20)
            self.evaluate_all_models()
            self.analyze_model_errors()

            # Step 7: Insights & Recommendations
            print("\n" + "=" * 17 + " STEP 7: INSIGHTS & RECOMMENDATIONS " + "=" * 17)
            self.generate_feature_insights()
            self.generate_business_recommendations()

            # Save artifacts
            print("\n" + "=" * 23 + " SAVING ARTIFACTS " + "=" * 23)
            self.save_model_artifacts()

            print("\n" + "=" * 60)
            print("🎉 COMPLETE ML PIPELINE EXECUTED SUCCESSFULLY! 🎉")
            print("=" * 60)

            # Final summary
            best_model_name = self.models[self.best_model].__class__.__name__
            best_auc = self.evaluation_results[self.best_model]['metrics']['roc_auc']

            print(f"\n📊 FINAL RESULTS:")
            print(f"   🏆 Best Model: {best_model_name}")
            print(f"   🎯 Best AUC Score: {best_auc:.4f}")
            print(f"   📈 Ready for Production Deployment!")

        except Exception as e:
            print(f"\n❌ Pipeline failed with error: {str(e)}")
            raise


# =========================================================================
# USAGE EXAMPLE AND MAIN EXECUTION
# =========================================================================

def main():
    """
    Main function to demonstrate the complete ML pipeline
    """
    print("🚀 Netflix Movie Preference Prediction - ML Pipeline")
    print("🎬 Building End-to-End Recommendation System")
    print("-" * 60)

    # Initialize pipeline
    netflix_ml = NetflixMLPipeline(random_state=42)

    # Run complete pipeline
    netflix_ml.run_complete_pipeline()

    # Additional analysis examples
    print("\n" + "=" * 60)
    print("📋 ADDITIONAL ANALYSIS EXAMPLES")
    print("=" * 60)

    # Example: Make predictions for new user-movie pairs
    print("\n🔮 Example: Making Predictions for New User-Movie Pairs")

    # Get sample data for prediction
    X, y = netflix_ml.prepare_ml_data()
    best_model = netflix_ml.models[netflix_ml.best_model]

    # Sample predictions
    sample_indices = np.random.choice(X.index, 5, replace=False)
    sample_X = X.loc[sample_indices]
    predictions = best_model.predict_proba(sample_X)[:, 1]

    print("Sample Predictions:")
    for i, (idx, prob) in enumerate(zip(sample_indices, predictions)):
        user_id = netflix_ml.interaction_data.loc[idx, 'user_id']
        movie_id = netflix_ml.interaction_data.loc[idx, 'movie_id']
        actual = y.loc[idx]
        print(
            f"   {i + 1}. User {user_id}, Movie {movie_id}: {prob:.1%} probability (Actual: {'Liked' if actual else 'Disliked'})")

    return netflix_ml


if __name__ == "__main__":
    # Execute the complete pipeline
    pipeline = main()