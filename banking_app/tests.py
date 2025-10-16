import unittest
import os
import json
from models import AccountService, LoanService, DB_FILE

# Use a temporary file for testing to avoid polluting the actual DB
TEST_DB_FILE = 'test_bank_data.json'


class TestBankingServices(unittest.TestCase):

    def setUp(self):
        """Setup fresh test environment before each test."""
        # Create a fresh, empty test DB file
        with open(TEST_DB_FILE, 'w') as f:
            json.dump({"accounts": [], "loan_rules": LoanService(DB_FILE).get_loan_rules()}, f, indent=2)

        self.account_service = AccountService(db_file=TEST_DB_FILE)
        self.loan_service = LoanService(db_file=TEST_DB_FILE)

    def tearDown(self):
        """Clean up the temporary test file after each test."""
        if os.path.exists(TEST_DB_FILE):
            os.remove(TEST_DB_FILE)

    def test_01_account_creation_savings(self):
        """Test creation of a Savings account."""
        account = self.account_service.create_account("Alice", "P1234567A", "Savings", None, 1000.00)
        self.assertIsNotNone(account)
        self.assertEqual(account['name'], "Alice")
        self.assertEqual(account['account_type'], "Savings")
        self.assertIn('BANK00', account['account_number'])
        self.assertIsNone(account['tan'])

    def test_02_account_creation_current(self):
        """Test creation of a Current account (with TAN)."""
        account = self.account_service.create_account("Bob", "B9876543C", "Current", "TAN777", 50000.00)
        self.assertIsNotNone(account)
        self.assertEqual(account['account_type'], "Current")
        self.assertEqual(account['tan'], "TAN777")

    def test_03_loan_check_no_account(self):
        """Test loan application when account does not exist."""
        result = self.loan_service.check_loan_eligibility("NONEXISTENT", "Personal Loan")
        self.assertEqual(result['status'], "REQUIRES_ACCOUNT")

    def test_04_loan_check_standard_approval(self):
        """Test standard loan approval for a Savings account."""
        savings_acc = self.account_service.create_account("Charlie", "C1111111C", "Savings", None, 5000.00)

        result = self.loan_service.check_loan_eligibility(savings_acc['account_number'], "Home Loan")
        self.assertEqual(result['status'], "APPROVED_FOR_PROCESSING")
        self.assertIn('loan_id', result)

    def test_05_loan_check_business_loan_savings_rejection(self):
        """Test business loan application by Savings account holder (initial request)."""
        savings_acc = self.account_service.create_account("David", "D2222222D", "Savings", None, 10000.00)

        result = self.loan_service.check_loan_eligibility(savings_acc['account_number'], "Business Loan")
        self.assertEqual(result['status'], "REQUIRES_UPGRADE")
        self.assertIn('required_action', result)

    def test_06_loan_check_business_loan_full_upgrade_flow(self):
        """Test the full workflow: Savings -> Upgrade -> Business Loan Approval."""
        savings_acc = self.account_service.create_account("Eve", "E3333333E", "Savings", None, 20000.00)
        acc_num = savings_acc['account_number']

        # 1. Initial attempt fails
        result_fail = self.loan_service.check_loan_eligibility(acc_num, "Business Loan")
        self.assertEqual(result_fail['status'], "REQUIRES_UPGRADE")

        # 2. Attempt upgrade with TAN
        upgrade_result = self.account_service.update_account_type(acc_num, 'Current', 'NEWTAN456')
        self.assertEqual(upgrade_result['account_type'], 'Current')
        self.assertEqual(upgrade_result['tan'], 'NEWTAN456')

        # 3. Second loan attempt (after successful upgrade)
        result_success = self.loan_service.check_loan_eligibility(acc_num, "Business Loan")
        self.assertEqual(result_success['status'], "APPROVED_FOR_PROCESSING")
        self.assertIn('loan_id', result_success)

    def test_07_loan_check_business_loan_current_account(self):
        """Test business loan approval for an existing Current account holder."""
        current_acc = self.account_service.create_account("Frank", "F4444444F", "Current", "TAN000", 99999.00)

        result = self.loan_service.check_loan_eligibility(current_acc['account_number'], "Business Loan")
        self.assertEqual(result['status'], "APPROVED_FOR_PROCESSING")


if __name__ == '__main__':
    # Add a simple text runner that captures output
    print("--- Running Unit Tests ---")
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(unittest.makeSuite(TestBankingServices))
