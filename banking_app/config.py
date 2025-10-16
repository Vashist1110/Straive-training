import os
from datetime import timedelta


class Config:
    # Secret key for JWT
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'

    # JWT Configuration
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY') or 'jwt-secret-key-change-in-production'
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=30)

    # Database Configuration
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
                              'sqlite:///loan_approval.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Pagination
    ITEMS_PER_PAGE = 10

    # Loan Configuration
    LOAN_TYPES = ['education', 'business', 'personal', 'car', 'home']
    ACCOUNT_TYPES = ['savings', 'current']

    # Minimum account balance requirements
    MIN_BALANCE = {
        'savings': 1000,
        'current': 5000
    }


class TestConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///test_loan_approval.db'
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(minutes=5)