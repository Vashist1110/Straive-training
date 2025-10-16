import unittest
from calculator import Calculator

class TestCalculator(unittest.TestCase):

    def setUp(self):
        self.calc = Calculator()

    def test_add(self):
        self.assertEqual(self.calc.add(5,10),15)
        self.assertEqual(self.calc.add(-1, -1), -2)
        self.assertEqual(self.calc.add(5, -1), 4)

    def test_sub(self):
        self.assertEqual(self.calc.subtract(10,5),5)
        self.assertEqual(self.calc.subtract(10,-5),15)
        self.assertEqual(self.calc.subtract(-5,-5),0)

    def test_mul(self):
        self.assertEqual(self.calc.multiply(5,2),10)

    def test_divide(self):
        self.assertEqual(self.calc.divide(10,5),2)

    def test_pow(self):
        self.assertEqual(self.calc.power(3,2),9)

    def test_mod(self):
        self.assertEqual(self.calc.modulus(7,3),1)

