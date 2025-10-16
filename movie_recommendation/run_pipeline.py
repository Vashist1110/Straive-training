#!/usr/bin/env python3
"""
Netflix ML Pipeline Runner
==========================

Simple script to run the complete Netflix movie preference prediction pipeline.
This script provides options for running different parts of the pipeline.

Usage:
    python run_pipeline.py --mode full              # Run complete pipeline
    python run_pipeline.py --mode quick             # Run with smaller dataset
    python run_pipeline.py --mode evaluation        # Only evaluation and insights
"""

import argparse
import sys
import time
from pathlib import Path

# Import the main pipeline
try:
    from netflix_ml_model import NetflixMLPipeline
except ImportError:
    print(" Error: netflix_ml_model.py not found in current directory")
    print("   Make sure netflix_ml_model.py is in the same folder")
    sys.exit(1)


def run_full_pipeline():
    """Run the complete ML pipeline"""
    print("🚀 Running Complete Netflix ML Pipeline")
    print("⏱️  This may take 10-15 minutes depending on your system...")

    start_time = time.time()

    # Initialize and run pipeline
    netflix_ml = NetflixMLPipeline(random_state=42)
    netflix_ml.run_complete_pipeline()

    end_time = time.time()
    duration = (end_time - start_time) / 60

    print(f"\n⏱️  Pipeline completed in {duration:.1f} minutes")
    return netflix_ml


def run_quick_pipeline():
    """Run a quick version with smaller dataset"""
    print("⚡ Running Quick Netflix ML Pipeline")
    print("📊 Using smaller dataset for faster execution...")

    start_time = time.time()

    # Initialize pipeline
    netflix_ml = NetflixMLPipeline(random_state=42)

    # Generate smaller dataset
    print("\n🎬 Generating smaller sample dataset...")
    netflix_ml.users_df = netflix_ml.users_df.iloc[:1000] if netflix_ml.users_df is not None else None
    netflix_ml.movies_df = netflix_ml.movies_df.iloc[:200] if netflix_ml.movies_df is not None else None

    # Generate data with smaller size
    netflix_ml.generate_sample_data()
    # Reduce interaction data
    netflix_ml.ratings_df = netflix_ml.ratings_df.sample(n=min(5000, len(netflix_ml.ratings_df)), random_state=42)

    # Run pipeline steps
    netflix_ml.handle_missing_values()
    netflix_ml.encode_categorical_features()
    netflix_ml.normalize_numerical_features()
    netflix_ml.create_smart_train_test_split()
    netflix_ml.create_complete_feature_pipeline()

    # Train only 2 models for speed
    X, y = netflix_ml.prepare_ml_data()
    train_idx = netflix_ml.train_data.index
    val_idx = netflix_ml.val_data.index

    X_train = X.loc[train_idx]
    y_train = y.loc[train_idx]
    X_val = X.loc[val_idx]
    y_val = y.loc[val_idx]

    print("\n🧠 Training models (Quick mode - 2 models only)...")
    netflix_ml.train_logistic_regression(X_train, y_train, X_val, y_val)
    netflix_ml.train_random_forest(X_train, y_train, X_val, y_val)

    # Evaluate
    netflix_ml.evaluate_all_models()
    netflix_ml.generate_feature_insights()

    end_time = time.time()
    duration = (end_time - start_time) / 60

    print(f"\n⏱️  Quick pipeline completed in {duration:.1f} minutes")
    return netflix_ml


def run_evaluation_only():
    """Run only evaluation and insights on pre-trained models"""
    print("📈 Running Evaluation and Insights Only")
    print("⚠️  This requires pre-trained models from a previous run")

    try:
        # Try to load existing models
        import pickle
        models_path = Path("netflix_models/")

        if not models_path.exists():
            print("❌ No pre-trained models found!")
            print("   Run with --mode full or --mode quick first")
            return None

        netflix_ml = NetflixMLPipeline(random_state=42)
        netflix_ml.generate_sample_data()  # Need data for evaluation
        netflix_ml.handle_missing_values()
        netflix_ml.encode_categorical_features()
        netflix_ml.normalize_numerical_features()
        netflix_ml.create_smart_train_test_split()
        netflix_ml.create_complete_feature_pipeline()

        # Load saved models
        model_files = {
            'logistic_regression': 'logistic_regression_model.pkl',
            'random_forest': 'random_forest_model.pkl',
            'xgboost': 'xgboost_model.pkl'
        }

        for model_key, filename in model_files.items():
            model_path = models_path / filename
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    netflix_ml.models[model_key] = pickle.load(f)
                print(f"✅ Loaded {model_key} model")

        if not netflix_ml.models:
            print("❌ No models could be loaded!")
            return None

        # Run evaluation
        netflix_ml.evaluate_all_models()
        netflix_ml.analyze_model_errors()
        netflix_ml.generate_feature_insights()
        netflix_ml.generate_business_recommendations()

        return netflix_ml

    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return None


def run_interactive_demo():
    """Run an interactive demonstration"""
    print("🎮 Interactive Netflix ML Demo")
    print("=" * 40)

    netflix_ml = NetflixMLPipeline(random_state=42)

    print("\n1. Generating sample data...")
    netflix_ml.generate_sample_data()

    print("\n2. Quick preprocessing...")
    netflix_ml.handle_missing_values()
    netflix_ml.encode_categorical_features()
    netflix_ml.normalize_numerical_features()

    print("\n3. Creating feature pipeline...")
    netflix_ml.create_smart_train_test_split()
    netflix_ml.create_complete_feature_pipeline()

    print("\n4. Training a quick model...")
    X, y = netflix_ml.prepare_ml_data()
    train_idx = netflix_ml.train_data.index
    val_idx = netflix_ml.val_data.index

    X_train = X.loc[train_idx]
    y_train = y.loc[train_idx]
    X_val = X.loc[val_idx]
    y_val = y.loc[val_idx]

    # Train just Random Forest for demo
    netflix_ml.train_random_forest(X_train, y_train, X_val, y_val)

    print("\n🎯 Making sample predictions...")
    model = netflix_ml.models['random_forest']

    # Get 5 random samples
    import numpy as np
    sample_indices = np.random.choice(X_train.index, 5, replace=False)
    sample_X = X.loc[sample_indices]
    predictions = model.predict_proba(sample_X)[:, 1]
    actual = y.loc[sample_indices]

    print("\n📊 Sample Predictions:")
    print("-" * 50)
    for i, (idx, prob, act) in enumerate(zip(sample_indices, predictions, actual)):
        user_id = netflix_ml.interaction_data.loc[idx, 'user_id']
        movie_id = netflix_ml.interaction_data.loc[idx, 'movie_id']
        movie_title = netflix_ml.movies_df.loc[netflix_ml.movies_df['movie_id'] == movie_id, 'title'].iloc[0]
        genre = netflix_ml.movies_df.loc[netflix_ml.movies_df['movie_id'] == movie_id, 'genre'].iloc[0]

        prediction_text = "👍 WILL LIKE" if prob > 0.5 else "👎 WON'T LIKE"
        actual_text = "👍 LIKED" if act == 1 else "👎 DISLIKED"
        confidence = prob if prob > 0.5 else (1 - prob)

        print(f"   {i + 1}. User {user_id:4d} × {movie_title} ({genre})")
        print(f"      Prediction: {prediction_text} ({confidence:.1%} confidence)")
        print(f"      Actual:     {actual_text}")
        print(f"      {'✅ CORRECT' if (prob > 0.5) == (act == 1) else '❌ WRONG'}")
        print()

    return netflix_ml


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Netflix Movie Preference Prediction ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --mode full        # Complete pipeline (10-15 min)
  python run_pipeline.py --mode quick       # Quick run with smaller data (2-3 min)
  python run_pipeline.py --mode evaluation  # Evaluation only (requires saved models)
  python run_pipeline.py --mode demo        # Interactive demo (1-2 min)
        """
    )

    parser.add_argument(
        '--mode',
        choices=['full', 'quick', 'evaluation', 'demo'],
        default='full',
        help='Pipeline execution mode (default: full)'
    )

    parser.add_argument(
        '--save-path',
        default='netflix_models/',
        help='Path to save/load models (default: netflix_models/)'
    )

    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random state for reproducibility (default: 42)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

    # Print header
    print("🎬" + "=" * 58 + "🎬")
    print("🎬 NETFLIX MOVIE PREFERENCE PREDICTION ML PIPELINE 🎬")
    print("🎬" + "=" * 58 + "🎬")
    print(f"Mode: {args.mode.upper()}")
    print(f"Random State: {args.random_state}")
    print("-" * 62)

    # Execute based on mode
    try:
        if args.mode == 'full':
            pipeline = run_full_pipeline()
        elif args.mode == 'quick':
            pipeline = run_quick_pipeline()
        elif args.mode == 'evaluation':
            pipeline = run_evaluation_only()
        elif args.mode == 'demo':
            pipeline = run_interactive_demo()
        else:
            print(f"❌ Unknown mode: {args.mode}")
            return 1

        if pipeline is None:
            print("❌ Pipeline execution failed!")
            return 1

        print("\n" + "🎉" * 20)
        print("🎉 PIPELINE EXECUTION COMPLETED SUCCESSFULLY! 🎉")
        print("🎉" * 20)

        # Final summary
        if hasattr(pipeline, 'models') and pipeline.models:
            print(f"\n📊 SUMMARY:")
            print(f"   Models Trained: {len(pipeline.models)}")
            if hasattr(pipeline, 'evaluation_results'):
                best_model_key = max(pipeline.evaluation_results.keys(),
                                     key=lambda k: pipeline.evaluation_results[k]['metrics']['roc_auc'])
                best_auc = pipeline.evaluation_results[best_model_key]['metrics']['roc_auc']
                print(f"   Best Model: {best_model_key.replace('_', ' ').title()}")
                print(f"   Best AUC: {best_auc:.4f}")

            print(f"   Dataset Size: {len(pipeline.ratings_df):,} interactions")
            print(f"   Features: {len(pipeline.feature_df.columns) - 1}")

        return 0

    except KeyboardInterrupt:
        print("\n⏹️  Pipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)