import unittest
import json
from app import create_app, db
from config import TestConfig
from models.user import User, Account
from utils.seed_data import seed_all


class AuthTestCase(unittest.TestCase):
    """Test cases for authentication"""

    def setUp(self):
        """Set up test client and database"""
        self.app = create_app(TestConfig)
        self.client = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()
        db.create_all()
        seed_all()

    def tearDown(self):
        """Clean up after tests"""
        db.session.remove()
        db.drop_all()
        self.app_context.pop()

    def test_register_user_savings_account(self):
        """Test user registration with savings account"""
        response = self.client.post('/api/auth/register',
                                    data=json.dumps({
                                        'name': 'Test User',
                                        'pan_number': 'ABCDE1234F',
                                        'user_id': 'test_user_123',
                                        'password': 'TestPass@123',
                                        'initial_amount': 10000,
                                        'account_type': 'savings'
                                    }),
                                    content_type='application/json'
                                    )

        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data)
        self.assertIn('access_token', data)
        self.assertIn('user', data)
        self.assertEqual(data['user']['name'], 'Test User')

    def test_register_user_current_account(self):
        """Test user registration with current account"""
        response = self.client.post('/api/auth/register',
                                    data=json.dumps({
                                        'name': 'Business User',
                                        'pan_number': 'FGHIJ5678K',
                                        'tan_number': 'DELA12345E',
                                        'user_id': 'business_user_123',
                                        'password': 'TestPass@123',
                                        'initial_amount': 50000,
                                        'account_type': 'current'
                                    }),
                                    content_type='application/json'
                                    )

        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data)
        self.assertIn('access_token', data)
        self.assertEqual(data['user']['account']['account_type'], 'current')

    def test_register_duplicate_user_id(self):
        """Test registration with duplicate user ID"""
        # First registration
        self.client.post('/api/auth/register',
                         data=json.dumps({
                             'name': 'Test User',
                             'pan_number': 'ABCDE1234F',
                             'user_id': 'test_user_123',
                             'password': 'TestPass@123',
                             'initial_amount': 10000,
                             'account_type': 'savings'
                         }),
                         content_type='application/json'
                         )

        # Duplicate registration
        response = self.client.post('/api/auth/register',
                                    data=json.dumps({
                                        'name': 'Another User',
                                        'pan_number': 'KLMNO9012P',
                                        'user_id': 'test_user_123',
                                        'password': 'TestPass@456',
                                        'initial_amount': 10000,
                                        'account_type': 'savings'
                                    }),
                                    content_type='application/json'
                                    )

        self.assertEqual(response.status_code, 409)

    def test_register_invalid_pan(self):
        """Test registration with invalid PAN"""
        response = self.client.post('/api/auth/register',
                                    data=json.dumps({
                                        'name': 'Test User',
                                        'pan_number': 'INVALID',
                                        'user_id': 'test_user_123',
                                        'password': 'TestPass@123',
                                        'initial_amount': 10000,
                                        'account_type': 'savings'
                                    }),
                                    content_type='application/json'
                                    )

        self.assertEqual(response.status_code, 400)

    def test_register_current_account_without_tan(self):
        """Test current account registration without TAN"""
        response = self.client.post('/api/auth/register',
                                    data=json.dumps({
                                        'name': 'Business User',
                                        'pan_number': 'FGHIJ5678K',
                                        'user_id': 'business_user_123',
                                        'password': 'TestPass@123',
                                        'initial_amount': 50000,
                                        'account_type': 'current'
                                    }),
                                    content_type='application/json'
                                    )

        self.assertEqual(response.status_code, 400)

    def test_register_insufficient_initial_amount(self):
        """Test registration with insufficient initial amount"""
        response = self.client.post('/api/auth/register',
                                    data=json.dumps({
                                        'name': 'Test User',
                                        'pan_number': 'ABCDE1234F',
                                        'user_id': 'test_user_123',
                                        'password': 'TestPass@123',
                                        'initial_amount': 500,  # Less than minimum
                                        'account_type': 'savings'
                                    }),
                                    content_type='application/json'
                                    )

        self.assertEqual(response.status_code, 400)

    def test_login_success(self):
        """Test successful login"""
        # Register user first
        self.client.post('/api/auth/register',
                         data=json.dumps({
                             'name': 'Test User',
                             'pan_number': 'ABCDE1234F',
                             'user_id': 'test_user_123',
                             'password': 'TestPass@123',
                             'initial_amount': 10000,
                             'account_type': 'savings'
                         }),
                         content_type='application/json'
                         )

        # Login
        response = self.client.post('/api/auth/login',
                                    data=json.dumps({
                                        'user_id': 'test_user_123',
                                        'password': 'TestPass@123'
                                    }),
                                    content_type='application/json'
                                    )

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('access_token', data)

    def test_login_invalid_credentials(self):
        """Test login with invalid credentials"""
        # Register user first
        self.client.post('/api/auth/register',
                         data=json.dumps({
                             'name': 'Test User',
                             'pan_number': 'ABCDE1234F',
                             'user_id': 'test_user_123',
                             'password': 'TestPass@123',
                             'initial_amount': 10000,
                             'account_type': 'savings'
                         }),
                         content_type='application/json'
                         )

        # Login with wrong password
        response = self.client.post('/api/auth/login',
                                    data=json.dumps({
                                        'user_id': 'test_user_123',
                                        'password': 'WrongPassword'
                                    }),
                                    content_type='application/json'
                                    )

        self.assertEqual(response.status_code, 401)

    def test_login_nonexistent_user(self):
        """Test login with non-existent user"""
        response = self.client.post('/api/auth/login',
                                    data=json.dumps({
                                        'user_id': 'nonexistent_user',
                                        'password': 'TestPass@123'
                                    }),
                                    content_type='application/json'
                                    )

        self.assertEqual(response.status_code, 401)


if __name__ == '__main__':
    unittest.main()