from abc import ABC, abstractmethod
from datetime import datetime
import random


# ========== BASE EMPLOYEE ==========
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


# ========== MANAGER ==========
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

        return True, f"Loan {loan_id} approved for Rs.{loan_amount:.2f} at {interest_rate}%"

    def view_approved_loans(self):
        print(f"\nLoans Approved by {self._name}")
        if not self._approved_loans:
            print("No loans approved yet")
        else:
            for loan in self._approved_loans:
                print(f"Loan ID: {loan['loan_id']} | Account: {loan['account_number']} | "
                      f"Amount: Rs.{loan['amount']:.2f} | Rate: {loan['interest_rate']}%")
        print("\n")
