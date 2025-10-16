import unittest
import json
from app import create_app, db
from config import TestConfig
from utils.seed_data import seed_all


class LoanTestCase(unittest.TestCase):
    """Test cases for loan operations"""

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
                             'initial_amount': 100000,
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

    def test_get_loan_types(self):
        """Test getting loan types"""
        response = self.client.get('/api/loan/types')

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('loan_types', data)
        self.assertEqual(len(data['loan_types']), 5)

    def test_apply_education_loan_success(self):
        """Test successful education loan application"""
        response = self.client.post('/api/loan/apply',
                                    data=json.dumps({
                                        'loan_type': 'education',
                                        'amount': 500000,
                                        'tenure_months': 60
                                    }),
                                    content_type='application/json',
                                    headers=self.headers
                                    )

        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data)
        self.assertIn('loan', data)
        self.assertEqual(data['loan']['loan_type'], 'education')
        self.assertEqual(data['loan']['status'], 'pending')

    def test_apply_business_loan_without_current_account(self):
        """Test business loan application without current account"""
        response = self.client.post('/api/loan/apply',
                                    data=json.dumps({
                                        'loan_type': 'business',
                                        'amount': 1000000,
                                        'tenure_months': 60
                                    }),
                                    content_type='application/json',
                                    headers=self.headers
                                    )

        self.assertEqual(response.status_code, 400)

    def test_apply_business_loan_with_current_account(self):
        """Test business loan application with current account"""
        # Register user with current account
        self.client.post('/api/auth/register',
                         data=json.dumps({
                             'name': 'Business User',
                             'pan_number': 'FGHIJ5678K',
                             'tan_number': 'DELA12345E',
                             'user_id': 'business_user_123',
                             'password': 'TestPass@123',
                             'initial_amount': 200000,
                             'account_type': 'current'
                         }),
                         content_type='application/json'
                         )

        # Login
        response = self.client.post('/api/auth/login',
                                    data=json.dumps({
                                        'user_id': 'business_user_123',
                                        'password': 'TestPass@123'
                                    }),
                                    content_type='application/json'
                                    )

        token = json.loads(response.data)['access_token']
        headers = {'Authorization': f'Bearer {token}'}

        # Apply for business loan
        response = self.client.post('/api/loan/apply',
                                    data=json.dumps({
                                        'loan_type': 'business',
                                        'amount': 1000000,
                                        'tenure_months': 60
                                    }),
                                    content_type='application/json',
                                    headers=headers
                                    )

        self.assertEqual(response.status_code, 201)

    def test_apply_loan_exceed_max_amount(self):
        """Test loan application exceeding maximum amount"""
        response = self.client.post('/api/loan/apply',
                                    data=json.dumps({
                                        'loan_type': 'personal',
                                        'amount': 1000000,  # Exceeds personal loan max
                                        'tenure_months': 60
                                    }),
                                    content_type='application/json',
                                    headers=self.headers
                                    )

        self.assertEqual(response.status_code, 400)

    def test_apply_loan_invalid_tenure(self):
        """Test loan application with invalid tenure"""
        response = self.client.post('/api/loan/apply',
                                    data=json.dumps({
                                        'loan_type': 'education',
                                        'amount': 500000,
                                        'tenure_months': 6  # Less than minimum
                                    }),
                                    content_type='application/json',
                                    headers=self.headers
                                    )

        self.assertEqual(response.status_code, 400)

    def test_check_eligibility_education_loan(self):
        """Test eligibility check for education loan"""
        response = self.client.get('/api/loan/eligibility/education',
                                   headers=self.headers
                                   )

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('eligible', data)
        self.assertTrue(data['eligible'])

    def test_check_eligibility_business_loan(self):
        """Test eligibility check for business loan"""
        response = self.client.get('/api/loan/eligibility/business',
                                   headers=self.headers
                                   )

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('eligible', data)
        self.assertFalse(data['eligible'])  # Should be false for savings account

    def test_get_my_loans(self):
        """Test getting user's loans"""
        # Apply for a loan first
        self.client.post('/api/loan/apply',
                         data=json.dumps({
                             'loan_type': 'personal',
                             'amount': 100000,
                             'tenure_months': 24
                         }),
                         content_type='application/json',
                         headers=self.headers
                         )

        # Get loans
        response = self.client.get('/api/loan/my-loans',
                                   headers=self.headers
                                   )

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('loans', data)
        self.assertEqual(len(data['loans']), 1)

    def test_apply_loan_without_account(self):
        """Test loan application without account"""
        # This test would require creating a user without account
        # which is not possible with current registration flow
        pass

    def test_apply_loan_invalid_type(self):
        """Test loan application with invalid type"""
        response = self.client.post('/api/loan/apply',
                                    data=json.dumps({
                                        'loan_type': 'invalid_type',
                                        'amount': 100000,
                                        'tenure_months': 24
                                    }),
                                    content_type='application/json',
                                    headers=self.headers
                                    )

        self.assertEqual(response.status_code, 400)


if __name__ == '__main__':
    unittest.main()