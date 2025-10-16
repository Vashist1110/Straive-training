import unittest
import json
from app import create_app, db
from config import TestConfig
from utils.seed_data import seed_all


class AccountTestCase(unittest.TestCase):
    """Test cases for account operations"""

    def setUp(self):
        """Set up test client and database"""
        self.app = create_app(TestConfig)
        self.client = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()
        db.create_all()
        seed_all()

        # Register and login a test user
        self.client.post('/api/auth/register',
                         data=json.dumps({
                             'name': 'Test User',
                             'pan_number': 'ABCDE1234F',
                             'user_id': 'test_user_123',
                             'password': 'TestPass@123',
                             'initial_amount': 50000,
                             'account_type': 'savings'
                         }),
                         content_type='application/json'
                         )

        response = self.client.post('/api/auth/login',
                                    data=json.dumps({
                                        'user_id': 'test_user_123',
                                        'password': 'TestPass@123'
                                    }),
                                    content_type='application/json'
                                    )

        self.token = json.loads(response.data)['access_token']
        self.headers = {'Authorization': f'Bearer {self.token}'}

    def tearDown(self):
        """Clean up after tests"""
        db.session.remove()
        db.drop_all()
        self.app_context.pop()

    def test_get_profile(self):
        """Test getting user profile"""
        response = self.client.get('/api/account/profile',
                                   headers=self.headers
                                   )

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('user', data)
        self.assertEqual(data['user']['user_id'], 'test_user_123')

    def test_get_balance(self):
        """Test getting account balance"""
        response = self.client.get('/api/account/balance',
                                   headers=self.headers
                                   )

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('balance', data)
        self.assertEqual(data['balance'], 50000)

    def test_deposit_success(self):
        """Test successful deposit"""
        response = self.client.post('/api/account/deposit',
                                    data=json.dumps({
                                        'amount': 10000
                                    }),
                                    content_type='application/json',
                                    headers=self.headers
                                    )

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['new_balance'], 60000)

    def test_deposit_negative_amount(self):
        """Test deposit with negative amount"""
        response = self.client.post('/api/account/deposit',
                                    data=json.dumps({
                                        'amount': -1000
                                    }),
                                    content_type='application/json',
                                    headers=self.headers
                                    )

        self.assertEqual(response.status_code, 400)

    def test_deposit_zero_amount(self):
        """Test deposit with zero amount"""
        response = self.client.post('/api/account/deposit',
                                    data=json.dumps({
                                        'amount': 0
                                    }),
                                    content_type='application/json',
                                    headers=self.headers
                                    )

        self.assertEqual(response.status_code, 400)

    def test_modify_account_type_to_current(self):
        """Test modifying account type from savings to current"""
        response = self.client.put('/api/account/modify-type',
                                   data=json.dumps({
                                       'new_account_type': 'current',
                                       'tan_number': 'DELA12345E'
                                   }),
                                   content_type='application/json',
                                   headers=self.headers
                                   )

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['account']['account_type'], 'current')

    def test_modify_account_type_without_tan(self):
        """Test modifying to current account without TAN"""
        response = self.client.put('/api/account/modify-type',
                                   data=json.dumps({
                                       'new_account_type': 'current'
                                   }),
                                   content_type='application/json',
                                   headers=self.headers
                                   )

        self.assertEqual(response.status_code, 400)

    def test_modify_account_type_insufficient_balance(self):
        """Test modifying account type with insufficient balance"""
        # Register user with low balance
        self.client.post('/api/auth/register',
                         data=json.dumps({
                             'name': 'Low Balance User',
                             'pan_number': 'FGHIJ5678K',
                             'user_id': 'low_balance_user',
                             'password': 'TestPass@123',
                             'initial_amount': 2000,
                             'account_type': 'savings'
                         }),
                         content_type='application/json'
                         )

        # Login
        response = self.client.post('/api/auth/login',
                                    data=json.dumps({
                                        'user_id': 'low_balance_user',
                                        'password': 'TestPass@123'
                                    }),
                                    content_type='application/json'
                                    )

        token = json.loads(response.data)['access_token']
        headers = {'Authorization': f'Bearer {token}'}

        # Try to modify account type
        response = self.client.put('/api/account/modify-type',
                                   data=json.dumps({
                                       'new_account_type': 'current',
                                       'tan_number': 'DELB12345F'
                                   }),
                                   content_type='application/json',
                                   headers=headers
                                   )

        self.assertEqual(response.status_code, 400)

    def test_modify_account_type_to_savings(self):
        """Test modifying account type from current to savings"""
        # Register user with current account
        self.client.post('/api/auth/register',
                         data=json.dumps({
                             'name': 'Current User',
                             'pan_number': 'KLMNO9012P',
                             'tan_number': 'DELC12345G',
                             'user_id': 'current_user',
                             'password': 'TestPass@123',
                             'initial_amount': 50000,
                             'account_type': 'current'
                         }),
                         content_type='application/json'
                         )

        # Login
        response = self.client.post('/api/auth/login',
                                    data=json.dumps({
                                        'user_id': 'current_user',
                                        'password': 'TestPass@123'
                                    }),
                                    content_type='application/json'
                                    )

        token = json.loads(response.data)['access_token']
        headers = {'Authorization': f'Bearer {token}'}

        # Modify to savings
        response = self.client.put('/api/account/modify-type',
                                   data=json.dumps({
                                       'new_account_type': 'savings'
                                   }),
                                   content_type='application/json',
                                   headers=headers
                                   )

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['account']['account_type'], 'savings')

    def test_get_profile_unauthorized(self):
        """Test getting profile without authorization"""
        response = self.client.get('/api/account/profile')

        self.assertEqual(response.status_code, 401)


if __name__ == '__main__':
    unittest.main()