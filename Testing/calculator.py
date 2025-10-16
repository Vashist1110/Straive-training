class Calculator:

    def add(self,a,b):
        return a+b

    def subtract(self,a,b):
        return a-b

    def multiply(self,a,b):
        return a*b

    def divide(self,a,b):
        if b==0:
            raise ValueError("Cannot divide by zero")
        return a/b

    def power(self,a,b):
        return a**b

    def modulus(self,a,b):
        return a%b

    def classify(self,num):
        if num%2==0:
            return "Even"
        elif num%3==0:
            return "Divisible by 3"
        else:
            return "Other"