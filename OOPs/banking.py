from datetime import datetime
from abc import ABC, abstractmethod
import random


class AccountNumberGenerator:
    _instance = None
    _counter = 1000

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def generate(self):
        self._counter += 1
        return f"ACC{self._counter:08d}"


class Account(ABC):

    def __init__(self, account_holder, initial_balance=0):
        self._account_number = AccountNumberGenerator().generate()
        self._account_holder = account_holder
        self._balance = initial_balance
        self._is_active = True
        self._transaction_history = []
        self._log_transaction("Account Created", initial_balance)

    # Getters
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

        transaction = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'type': transaction_type,
            'amount': amount,
            'balance_after': balance_after
        }
        self._transaction_history.append(transaction)

    def deposit(self, amount):

        if not self._is_active:
            return False, "Account is inactive"

        if amount <= 0:
            return False, "Deposit amount must be positive"

        self._balance += amount
        self._log_transaction("Deposit", amount)
        return True, f"Deposited Rs.{amount:.2f} successfully"

    @abstractmethod
    def withdraw(self, amount):
        pass

    def transfer(self, amount, target_account):
        if not self._is_active:
            return False, "Source account is inactive"

        if not target_account.is_active:
            return False, "Target account is inactive"

        if amount <= 0:
            return False, "Transfer amount must be positive"

        # Attempt withdrawal from source
        success, message = self.withdraw(amount)
        if not success:
            return False, f"Transfer failed: {message}"

        # Deposit to target
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
        print("\n")
        print(f"Account Number: {self._account_number}")
        print(f"Account Holder: {self._account_holder}")
        print(f"Account Type: {self.__class__.__name__}")
        print(f"Balance: ${self._balance:.2f}")
        print(f"Status: {status}")
        self._display_specific_details()
        print("\n")

    def _display_specific_details(self):
        pass

    def display_transaction_history(self):

        print(f"\nTransaction History for {self._account_number}")
        if not self._transaction_history:
            print("No transactions yet")
        else:
            for trans in self._transaction_history:
                print(f"{trans['timestamp']} | {trans['type']:20s} | "
                      f"${trans['amount']:10.2f} | Balance: ${trans['balance_after']:.2f}")
        print("\n")


class SavingsAccount(Account):

    INTEREST_RATE = 0.025

    def withdraw(self, amount):

        if not self._is_active:
            return False, "Account is inactive"

        if amount <= 0:
            return False, "Withdrawal amount must be positive"

        if self._balance - amount < 0:
            return False, f"Insufficient funds. Balance: ${self._balance:.2f}"

        self._balance -= amount
        self._log_transaction("Withdrawal", -amount)
        return True, f"Withdrew Rs.{amount:.2f} successfully"

    def calculate_interest(self):

        if not self._is_active:
            return False, "Account is inactive"

        interest = self._balance * self.INTEREST_RATE
        self._balance += interest
        self._log_transaction("Interest Credit", interest)
        return True, f"Interest ${interest:.2f} credited at {self.INTEREST_RATE * 100}%"

    def _display_specific_details(self):
        print(f"Interest Rate: {self.INTEREST_RATE * 100}% annually")


class CurrentAccount(Account):

    def __init__(self, account_holder, initial_balance=0):
        super().__init__(account_holder, initial_balance)
        self._overdraft_limit = 500

    @property
    def overdraft_limit(self):
        return self._overdraft_limit

    def withdraw(self, amount):
        if not self._is_active:
            return False, "Account is inactive"

        if amount <= 0:
            return False, "Withdrawal amount must be positive"

        available_funds = self._balance + self._overdraft_limit
        if amount > available_funds:
            return False, f"Exceeds overdraft limit. Available: ${available_funds:.2f}"

        self._balance -= amount
        self._log_transaction("Withdrawal", -amount)
        return True, f"Withdrew ${amount:.2f} successfully"

    def _display_specific_details(self):
        print(f"Overdraft Limit: ${self._overdraft_limit:.2f}")
        available = self._balance + self._overdraft_limit
        print(f"Available Funds: ${available:.2f}")


class Employee(ABC):

    def __init__(self, emp_id, name, designation):
        self._emp_id = emp_id
        self._name = name
        self._designation = designation

    @property
    def name(self):
        return self._name

    @property
    def designation(self):
        return self._designation

    @abstractmethod
    def get_permissions(self):
        pass


class Manager(Employee):

    def __init__(self, emp_id, name):
        super().__init__(emp_id, name, "Manager")
        self._approved_loans = []

    def get_permissions(self):
        return ["approve_loans", "view_all_accounts", "manage_employees"]

    def approve_loan(self, account, loan_amount, interest_rate=5.0):
        if not account.is_active:
            return False, "Cannot approve loan for inactive account"

        if loan_amount <= 0:
            return False, "Loan amount must be positive"

        loan_id = f"LOAN{random.randint(10000, 99999)}"
        loan_details = {
            'loan_id': loan_id,
            'account_number': account.account_number,
            'amount': loan_amount,
            'interest_rate': interest_rate,
            'approved_by': self._name,
            'approved_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        self._approved_loans.append(loan_details)
        account.deposit(loan_amount)

        return True, f"Loan {loan_id} approved for ${loan_amount:.2f} at {interest_rate}%"

    def view_approved_loans(self):
        print(f"\nLoans Approved by {self._name}")
        if not self._approved_loans:
            print("No loans approved yet")
        else:
            for loan in self._approved_loans:
                print(f"Loan ID: {loan['loan_id']} | Account: {loan['account_number']} | "
                      f"Amount: ${loan['amount']:.2f} | Rate: {loan['interest_rate']}%")
        print("\n")


# ============= BANK SYSTEM =============
class BankSystem:
    """Main banking system to manage accounts"""

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
                account = CurrentAccount(account_holder, initial_balance, overdraft)
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
        """Get all managers"""
        return self._managers


# ============= MENU-DRIVEN INTERFACE =============
def display_menu():
    print("\n")
    print("    BANKING SYSTEM MENU")
    print("1.  Create New Account")
    print("2.  Deposit Money")
    print("3.  Withdraw Money")
    print("4.  Transfer Money")
    print("5.  View Account Details")
    print("6.  View Transaction History")
    print("7.  Calculate Interest (Savings Only)")
    print("8.  Close Account")
    print("9.  List All Accounts")
    print("10. Manager Operations")
    print("11. Exit")


def manager_menu(bank_system):
    """Manager operations menu"""
    print("\n")
    print("MANAGER OPERATIONS")
    print("1. Add Manager")
    print("2. Approve Loan")
    print("3. View Approved Loans")
    print("4. Back to Main Menu")

    choice = input("\nEnter choice: ")

    if choice == '1':
        emp_id = input("Enter Employee ID: ")
        name = input("Enter Manager Name: ")
        manager = bank_system.add_manager(emp_id, name)
        print(f"Manager {name} added successfully")

    elif choice == '2':
        managers = bank_system.get_managers()
        if not managers:
            print("No managers in system. Please add a manager first.")
            return

        print("\nAvailable Managers:")
        for i, mgr in enumerate(managers, 1):
            print(f"{i}. {mgr.name}")

        mgr_idx = int(input("Select manager: ")) - 1
        if 0 <= mgr_idx < len(managers):
            acc_num = input("Enter account number: ")
            account = bank_system.find_account(acc_num)
            if account:
                amount = float(input("Enter loan amount: $"))
                rate = float(input("Enter interest rate %: "))
                success, msg = managers[mgr_idx].approve_loan(account, amount, rate)
                print(msg)
            else:
                print("Account not found")

    elif choice == '3':
        managers = bank_system.get_managers()
        if not managers:
            print("No managers in system")
            return

        print("\nSelect Manager:")
        for i, mgr in enumerate(managers, 1):
            print(f"{i}. {mgr.name}")

        mgr_idx = int(input("Select manager: ")) - 1
        if 0 <= mgr_idx < len(managers):
            managers[mgr_idx].view_approved_loans()


def main():
    bank = BankSystem()

    # Create some sample data
    print("Initializing Banking System...")
    print("Sample accounts created for testing")

    while True:
        display_menu()
        choice = input("\nEnter your choice: ")

        try:
            if choice == '1':
                # Create account
                holder = input("Enter account holder name: ")
                acc_type = input("Enter account type (savings/current): ")
                balance = float(input("Enter initial balance: $"))

                if acc_type.lower() == 'current':
                    overdraft = float(input("Enter overdraft limit: $"))
                    account, msg = bank.create_account(acc_type, holder, balance, overdraft_limit=overdraft)
                else:
                    account, msg = bank.create_account(acc_type, holder, balance)

                print(msg)
                if account:
                    print(f"Your account number is: {account.account_number}")

            elif choice == '2':
                # Deposit
                acc_num = input("Enter account number: ")
                account = bank.find_account(acc_num)
                if account:
                    amount = float(input("Enter deposit amount: $"))
                    success, msg = account.deposit(amount)
                    print(msg)
                else:
                    print("Account not found")

            elif choice == '3':
                # Withdraw
                acc_num = input("Enter account number: ")
                account = bank.find_account(acc_num)
                if account:
                    amount = float(input("Enter withdrawal amount: $"))
                    success, msg = account.withdraw(amount)
                    print(msg)
                else:
                    print("Account not found")

            elif choice == '4':
                # Transfer
                src_num = input("Enter source account number: ")
                src_account = bank.find_account(src_num)
                if src_account:
                    tgt_num = input("Enter target account number: ")
                    tgt_account = bank.find_account(tgt_num)
                    if tgt_account:
                        amount = float(input("Enter transfer amount: $"))
                        success, msg = src_account.transfer(amount, tgt_account)
                        print(msg)
                    else:
                        print("Target account not found")
                else:
                    print("Source account not found")

            elif choice == '5':
                # View details
                acc_num = input("Enter account number: ")
                account = bank.find_account(acc_num)
                if account:
                    account.display_details()
                else:
                    print("Account not found")

            elif choice == '6':
                # Transaction history
                acc_num = input("Enter account number: ")
                account = bank.find_account(acc_num)
                if account:
                    account.display_transaction_history()
                else:
                    print("Account not found")

            elif choice == '7':
                # Calculate interest
                acc_num = input("Enter account number: ")
                account = bank.find_account(acc_num)
                if account:
                    if isinstance(account, SavingsAccount):
                        success, msg = account.calculate_interest()
                        print(msg)
                    else:
                        print("Interest calculation only for Savings Accounts")
                else:
                    print("Account not found")

            elif choice == '8':
                # Close account
                acc_num = input("Enter account number: ")
                account = bank.find_account(acc_num)
                if account:
                    success, msg = account.close_account()
                    print(msg)
                else:
                    print("Account not found")

            elif choice == '9':
                # List all accounts
                accounts = bank.get_all_accounts()
                if accounts:
                    print(f"\n{'=' * 70}")
                    print(f"{'Acc Number':<15} {'Holder':<20} {'Type':<15} {'Balance':>10}")
                    print(f"{'=' * 70}")
                    for acc in accounts:
                        status = "Active" if acc.is_active else "Inactive"
                        print(f"{acc.account_number:<15} {acc.account_holder:<20} "
                              f"{acc.__class__.__name__:<15} Rs.{acc.balance:>9.2f} ({status})")
                    print(f"{'=' * 70}\n")
                else:
                    print("No accounts in system")

            elif choice == '10':
                # Manager operations
                manager_menu(bank)

            elif choice == '11':
                # Exit
                print("\nThank you for using Banking System!")
                break

            else:
                print("Invalid choice. Please try again.")

        except ValueError as e:
            print(f"Invalid input: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")


# ============= SCALABILITY NOTES =============
"""
SCALABILITY CONSIDERATIONS:

1. DATABASE INTEGRATION:
   - Replace in-memory dictionary with database (PostgreSQL/MySQL)
   - Use ORM like SQLAlchemy for object-relational mapping
   - Implement connection pooling for concurrent requests

2. CACHING:
   - Use Redis for frequently accessed account data
   - Cache account balances and recent transactions
   - Implement cache invalidation strategies

3. MICROSERVICES ARCHITECTURE:
   - Separate services: Account Service, Transaction Service, Loan Service
   - Use message queues (RabbitMQ/Kafka) for async operations
   - Implement API Gateway for routing

4. CONCURRENCY & LOCKING:
   - Implement pessimistic/optimistic locking for transactions
   - Use database transactions with ACID properties
   - Implement distributed locks (Redis) for critical sections

5. LOAD BALANCING:
   - Use load balancers (NGINX, HAProxy) to distribute requests
   - Implement horizontal scaling with multiple app servers
   - Use auto-scaling based on metrics

6. DATA PARTITIONING:
   - Shard data by account number ranges
   - Use database replication for read scalability
   - Implement master-slave or multi-master setup

7. SECURITY:
   - Implement authentication (JWT tokens)
   - Use encryption for sensitive data
   - Implement rate limiting and DDoS protection
   - Audit logging for all transactions

8. MONITORING & OBSERVABILITY:
   - Implement logging (ELK stack)
   - Use metrics (Prometheus, Grafana)
   - Distributed tracing (Jaeger, Zipkin)
   - Health checks and alerting

9. API DESIGN:
   - RESTful APIs with proper versioning
   - GraphQL for flexible queries
   - WebSockets for real-time updates

10. BACKUP & DISASTER RECOVERY:
    - Regular automated backups
    - Point-in-time recovery
    - Multi-region deployment for high availability
"""

if __name__ == "__main__":
    main()