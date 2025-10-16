import pytest
from emi_calc import Account, calculate_emi


@pytest.mark.parametrize("principal, annual_rate, years", [
    (100000, 0.1, 1),
    (100000, 0.1, 5),
    (100000, 0.1, 30),
])
def test_calculate_emi(principal, annual_rate, years):
    emi = calculate_emi(principal, annual_rate, years)
    n = years * 12

    total_paid = round(emi * n, 2)
    interest_paid = total_paid - principal

    assert total_paid > principal
    assert interest_paid > 0

def test_transaction_history_updates():
    acc = Account("Test", 0)

    acc.deposit(1000)
    acc.withdraw(200)
    acc.deposit(500)

    assert len(acc.history) == 3

    assert acc.history[-1][2] == acc.balance

    assert acc.history[0] == ("deposit", 1000, 1000)
    assert acc.history[1] == ("withdraw", 200, 800)
    assert acc.history[2] == ("deposit", 500, 1300)

def test_full_year_simulation():
    acc = Account("Alice", 20000, 0.06)

    for _ in range(12):
        acc.deposit(500)

    interest = acc.calculate_annual_interest()
    acc.deposit(interest)

    acc.withdraw(3000)

    expected_balance = 24560

    assert acc.balance == expected_balance

def test_zero_interest_no_growth():
    acc = Account("ZeroInterest", 10000, 0.0)

    for _ in range(12):
        acc.deposit(1000)

    interest = acc.calculate_annual_interest()
    assert interest == 0.0

    acc.balance += interest
    assert acc.balance == 10000 + 12 * 1000

def test_negative_interest_penalty():
    acc = Account("PenaltyAccount", 10000, -0.05)

    for _ in range(12):
        acc.deposit(1000)

    interest = acc.calculate_annual_interest()
    assert interest < 0


    acc.balance += interest
    expected_balance = 10000 + 12000 + interest
    assert acc.balance == expected_balance

