import pytest
from interest import Account, InsufficientbalanceError


@pytest.fixture
def accounts():
    alice = Account("Alice", 0)
    bob = Account("Bob", 0)
    return alice, bob


def test_alice_transactions(accounts):
    alice,bob = accounts

    # Track debits and credits for Alice
    alice_credits = 0
    alice_debits = 0

    alice.deposit(2000)
    alice_credits += 2000

    alice.withdraw(500)
    alice_debits += 500

    alice.transfer(bob, 1000)
    alice_debits += 1000


    assert alice.balance == 500
    assert bob.balance == 1000

    assert alice_credits == 2000
    assert alice_debits == 1500




# def test_deposit_valid(account):
#     account.deposit(500)
#     assert account.balance == 1500
#
# @pytest.mark.parametrize("invalid_amount", [0, -100, -1])
# def test_deposit_invalid_raises(account, invalid_amount):
#     with pytest.raises(ValueError, match="Deposit must be positive"):
#         account.deposit(invalid_amount)
#
# def test_withdraw_valid(account):
#     account.withdraw(400)
#     assert account.balance == 600
#
# def test_withdraw_insufficient_balance_raises(account):
#     with pytest.raises(InsufficientbalanceError, match="Not Enough balance"):
#         account.withdraw(1500)
#
# def test_calculate_annual_interest(account):
#     expected_interest = round(account.balance * account.annual_rate, 2)
#     assert account.calculate_annual_interest() == expected_interest
#
# def test_calculate_compound_interest(account):
#     expected_amount = round(account.balance * ((1 + account.annual_rate) ** 2), 2)
#     assert account.calculate_compound_interest(2) == expected_amount
#
# def test_calculate_compound_interest_with_frequency(account):
#     p = account.balance
#     r = account.annual_rate
#     n = 4
#     t = 1
#     expected_amount = round(p * ((1 + r/n) ** (n*t)), 2)
#     assert account.calculate_compound_interest(t, compounding_freq=n) == expected_amount
#
# @pytest.mark.parametrize("years", [1, 2, 5, 10])
# def test_calculate_compound_interest_years(account, years):
#     p = account.balance
#     r = account.annual_rate
#     n = 1
#     t = years
#     expected_amount = round(p * ((1 + r/n) ** (n*t)), 2)
#     assert account.calculate_compound_interest(years) == expected_amount

