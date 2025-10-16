from accounts import SavingsAccount, CurrentAccount
from employees import Manager


class BankSystem:
    """Main banking system to manage accounts and managers"""

    def __init__(self):
        self._accounts = {}  # Dictionary for O(1) lookup
        self._managers = []

    def create_account(self, account_type, account_holder, initial_balance=0, **kwargs):
        """Create a new account"""

        try:
            if account_type.lower() == 'savings':
                account = SavingsAccount(account_holder, initial_balance)

            elif account_type.lower() == 'current':
                overdraft = kwargs.get('overdraft_limit', 500)
                account = CurrentAccount(account_holder, initial_balance)
                account._overdraft_limit = overdraft  # Customize overdraft if needed

            else:
                return None, "Invalid account type"

            self._accounts[account.account_number] = account
            return account, f"Account {account.account_number} created successfully"

        except Exception as e:
            return None, f"Error creating account: {str(e)}"

    def find_account(self, account_number):
        """Search account by account number - O(1) complexity"""
        return self._accounts.get(account_number)

    def get_all_accounts(self):
        """Return all accounts"""
        return list(self._accounts.values())

    def add_manager(self, emp_id, name):
        """Add a manager to the system"""
        manager = Manager(emp_id, name)
        self._managers.append(manager)
        return manager

    def get_managers(self):
        """Return list of all managers"""
        return self._managers
