from models.user import Account
from app import db
from utils.exceptions import ValidationError
from config import Config


class AccountService:
    """Service class for account operations"""

    def modify_account_type(self, account, new_account_type):
        """
        Modify account type

        Args:
            account: Account object
            new_account_type: New account type (savings/current)

        Returns:
            Account: Updated account object

        Raises:
            ValidationError: If validation fails
        """
        if account.account_type == new_account_type:
            raise ValidationError(f'Account is already of type {new_account_type}')

        # Check minimum balance requirement
        min_balance = Config.MIN_BALANCE.get(new_account_type, 0)
        if account.balance < min_balance:
            raise ValidationError(
                f'Insufficient balance. Minimum balance of {min_balance} required for {new_account_type} account'
            )

        account.account_type = new_account_type

        return account

    def deposit(self, account, amount):
        """
        Deposit money to account

        Args:
            account: Account object
            amount: Amount to deposit

        Returns:
            Account: Updated account object

        Raises:
            ValidationError: If amount is invalid
        """
        if amount <= 0:
            raise ValidationError('Amount must be greater than 0')

        account.balance += amount
        return account

    def withdraw(self, account, amount):
        """
        Withdraw money from account

        Args:
            account: Account object
            amount: Amount to withdraw

        Returns:
            Account: Updated account object

        Raises:
            ValidationError: If amount is invalid or insufficient balance
        """
        if amount <= 0:
            raise ValidationError('Amount must be greater than 0')

        min_balance = Config.MIN_BALANCE.get(account.account_type, 0)

        if account.balance - amount < min_balance:
            raise ValidationError(
                f'Insufficient balance. Minimum balance of {min_balance} must be maintained'
            )

        account.balance -= amount
        return account