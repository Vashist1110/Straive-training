import pytest
from calculator import Calculator

@pytest.fixture()

def calc():
    return Calculator()
def test_add(calc):
    assert calc.add(5,10)==15

def test_sub(calc):
    assert calc.subtract(5,10)==-5

def test_mul(calc):
    assert calc.multiply(2,3)==6
def test_divide(calc):
    assert calc.divide(10,5)==2

def test_classify(calc):
    assert calc.classify(15) == "Divisible by 3"
    assert calc.classify(10) == "Even"
    assert calc.classify(5) == "Other"
