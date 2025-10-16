from datetime import datetime
from abc import ABC, abstractmethod

class AccountNumberGenerator:
    _instance = None
    _counter = 1000

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def generate(self):
        self._counter += 1
        return f"{self._counter:08d}"


class Account(ABC):
    def __init__(self, account_holder, initial_balance=0):
        self._account_number = AccountNumberGenerator().generate()
        self._account_holder = account_holder
        self._balance = initial_balance
        self._is_active = True
        self._transaction_history = []
        self._log_transaction("Account Created", initial_balance)

    @property
    def account_number(self):
        return self._account_number

    @property
    def account_holder(self):
        return self._account_holder

    @property
    def balance(self):
        return self._balance

    @property
    def is_active(self):
        return self._is_active

    def _log_transaction(self, transaction_type, amount, balance_after=None):
        if balance_after is None:
            balance_after = self._balance
        self._transaction_history.append({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'type': transaction_type,
            'amount': amount,
            'balance_after': balance_after
        })

    def deposit(self, amount):
        if not self._is_active:
            return False, "Account is inactive"
        if amount <= 0:
            return False, "Deposit amount must be positive"
        self._balance += amount
        self._log_transaction("Deposit", amount)
        return True, f"Deposited Rs.{amount:.2f} successfully"

    def transfer(self, amount, target_account):
        if not self._is_active:
            return False, "Source account is inactive"
        if not target_account.is_active:
            return False, "Target account is inactive"
        if amount <= 0:
            return False, "Transfer amount must be positive"

        success, message = self.withdraw(amount)
        if not success:
            return False, f"Transfer failed: {message}"

        target_account.deposit(amount)
        self._log_transaction(f"Transfer to {target_account.account_number}", -amount)
        target_account._log_transaction(f"Transfer from {self.account_number}", amount)
        return True, f"Transferred Rs.{amount:.2f} to {target_account.account_number}"

    def close_account(self):
        if self._balance > 0:
            return False, f"Cannot close account with balance Rs.{self._balance:.2f}"
        self._is_active = False
        self._log_transaction("Account Closed", 0)
        return True, "Account closed successfully"

    def get_transaction_history(self):
        return self._transaction_history.copy()

    def display_details(self):
        status = "Active" if self._is_active else "Inactive"
        print(f"\nAccount Number: {self._account_number}")
        print(f"Account Holder: {self._account_holder}")
        print(f"Account Type: {self.__class__.__name__}")
        print(f"Balance: Rs.{self._balance:.2f}")
        print(f"Status: {status}")
        self._display_specific_details()

    def display_transaction_history(self):
        print(f"\nTransaction History for {self._account_number}")
        for trans in self._transaction_history:
            print(f"{trans['timestamp']} | {trans['type']:20s} | Rs.{trans['amount']:10.2f} | Balance: Rs.{trans['balance_after']:.2f}")
        print("\n")

    def _display_specific_details(self):
        pass

    @abstractmethod
    def withdraw(self, amount):
        pass


class SavingsAccount(Account):
    INTEREST_RATE = 0.04

    def withdraw(self, amount):
        if not self._is_active:
            return False, "Account is inactive"
        if amount <= 0:
            return False, "Withdrawal amount must be positive"
        if self._balance - amount < 0:
            return False, f"Insufficient funds. Balance: Rs.{self._balance:.2f}"
        self._balance -= amount
        self._log_transaction("Withdrawal", -amount)
        return True, f"Withdrew Rs.{amount:.2f} successfully"

    def calculate_interest(self):
        if not self._is_active:
            return False, "Account is inactive"
        interest = self._balance * self.INTEREST_RATE
        self._balance += interest
        self._log_transaction("Interest Credit", interest)
        return True, f"Interest Rs.{interest:.2f} credited at {self.INTEREST_RATE * 100}%"

    def _display_specific_details(self):
        print(f"Interest Rate: {self.INTEREST_RATE * 100}%")


class CurrentAccount(Account):
    def __init__(self, account_holder, initial_balance=0):
        super().__init__(account_holder, initial_balance)
        self._overdraft_limit = 5000

    @property
    def overdraft_limit(self):
        return self._overdraft_limit

    def withdraw(self, amount):
        if not self._is_active:
            return False, "Account is inactive"
        if amount <= 0:
            return False, "Withdrawal amount must be positive"
        if amount > self._balance + self._overdraft_limit:
            return False, f"Exceeds overdraft limit. Available: Rs.{self._balance + self._overdraft_limit:.2f}"
        self._balance -= amount
        self._log_transaction("Withdrawal", -amount)
        return True, f"Withdrew Rs.{amount:.2f} successfully"

    def _display_specific_details(self):
        print(f"Overdraft Limit: Rs.{self._overdraft_limit:.2f}")
        print(f"Available Funds: Rs.{self._balance + self._overdraft_limit:.2f}")
