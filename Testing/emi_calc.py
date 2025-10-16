class InsufficientbalanceError(Exception):
    pass

class Account:
    def __init__(self, owner, balance=0, annual_rate=0.05):
        """
        :param owner: Account holder name
        :param balance: Initial balance
        :param annual_rate: Annual interest rate (e.g., 0.05 for 5%)
        """
        self.owner = owner
        self.balance = balance
        self.annual_rate = annual_rate
        self.history = []

    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("Deposit must be positive")
        self.balance += amount
        self.history.append(("deposit", amount, self.balance))
        return self.balance

    def withdraw(self, amount):
        if amount > self.balance:
            raise InsufficientbalanceError("Not Enough balance")
        self.balance -= amount
        self.history.append(("withdraw", amount, self.balance))
        return self.balance

    def transfer(self, target_account, amount):
        self.withdraw(amount)
        target_account.deposit(amount)
        return self.balance, target_account.balance

    def calculate_annual_interest(self):
        """
        Formula: Interest = balance * annual_rate
        """
        return round(self.balance * self.annual_rate, 2)

    def calculate_compound_interest(self, years, compounding_freq=1):
        p = self.balance
        r = self.annual_rate
        n = compounding_freq
        t = years
        a = p * ((1 + r / n) ** (n * t))
        return round(a, 2)


def calculate_emi(principal, annual_rate, years):
    """
    Calculate EMI (Equated Monthly Installment) for a loan.

    :param principal: Loan amount
    :param annual_rate: Annual interest rate (e.g., 0.1 for 10%)
    :param years: Number of years for loan repayment
    :return: EMI amount rounded to 2 decimals
    """
    r = annual_rate / 12
    n = years * 12
    emi = (principal * r * (1 + r) ** n) / ((1 + r) ** n - 1)
    return round(emi, 2)
