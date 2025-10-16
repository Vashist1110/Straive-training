import pytest
from bankingapp import Account, InsufficientBalanceError  # Replace 'your_module' with your file name


@pytest.fixture
def accounts():
    acc1 = Account("Alice", 1000)
    acc2 = Account("Bob", 500)
    return acc1, acc2


def test_deposit_increases_balance(accounts):
    acc1, _ = accounts
    acc1.deposit(500)
    assert acc1.balance == 1500

@pytest.mark.parametrize("invalid_amount", [0, -100, -1])
def test_deposit_invalid_values_raise(accounts, invalid_amount):
    acc1, _ = accounts
    with pytest.raises(ValueError):
        acc1.deposit(invalid_amount)



def test_withdraw_decreases_balance(accounts):
    acc1, _ = accounts
    acc1.withdraw(400)
    assert acc1.balance == 600

def test_withdraw_more_than_balance_raises(accounts):
    acc1, _ = accounts
    with pytest.raises(InsufficientBalanceError):
        acc1.withdraw(1500)



def test_transfer_money_updates_balances(accounts):
    acc1, acc2 = accounts
    acc1.transfer(acc2, 300)
    assert acc1.balance == 700
    assert acc2.balance == 800

def test_transfer_fails_with_insufficient_funds(accounts):
    acc1, acc2 = accounts
    with pytest.raises(InsufficientBalanceError):
        acc1.transfer(acc2, 1500)

    assert acc1.balance == 1000
    assert acc2.balance == 500



@pytest.mark.parametrize("withdraw_amount", [100, 500, 1000])
def test_withdraw_various_amounts(accounts, withdraw_amount):
    acc1, _ = accounts
    if withdraw_amount <= acc1.balance:
        old_balance = acc1.balance
        acc1.withdraw(withdraw_amount)
        assert acc1.balance == old_balance - withdraw_amount
    else:
        with pytest.raises(InsufficientBalanceError):
            acc1.withdraw(withdraw_amount)



def test_withdraw_exception_message(accounts):
    acc1, _ = accounts
    with pytest.raises(InsufficientBalanceError) as exc_info:
        acc1.withdraw(1500)
    assert str(exc_info.value) == "Not enough balance"



def test_transfer_full_balance(accounts):
    acc1, acc2 = accounts
    full_balance = acc1.balance
    acc1.transfer(acc2, full_balance)
    assert acc1.balance == 0
    assert acc2.balance == 500 + full_balance
