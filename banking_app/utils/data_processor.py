import pandas as pd
from models.loan import Loan, LoanEligibility
from models.user import User, Account
from app import db
from datetime import datetime


class DataProcessor:
    """Class for processing and analyzing data using pandas"""

    @staticmethod
    def export_loans_to_dataframe():
        """
        Export all loans to pandas DataFrame

        Returns:
            pd.DataFrame: DataFrame containing loan data
        """
        loans = Loan.query.all()

        data = []
        for loan in loans:
            data.append({
                'loan_id': loan.id,
                'loan_number': loan.loan_number,
                'user_id': loan.user.user_id,
                'user_name': loan.user.name,
                'loan_type': loan.loan_type,
                'amount': loan.amount,
                'interest_rate': loan.interest_rate,
                'tenure_months': loan.tenure_months,
                'status': loan.status,
                'application_date': loan.application_date,
                'approval_date': loan.approval_date,
                'account_type': loan.user.account.account_type if loan.user.account else None,
                'account_balance': loan.user.account.balance if loan.user.account else None
            })

        return pd.DataFrame(data)

    @staticmethod
    def export_users_to_dataframe():
        """
        Export all users to pandas DataFrame

        Returns:
            pd.DataFrame: DataFrame containing user data
        """
        users = User.query.all()

        data = []
        for user in users:
            data.append({
                'user_id': user.user_id,
                'name': user.name,
                'pan_number': user.pan_number,
                'tan_number': user.tan_number,
                'has_account': user.account is not None,
                'account_type': user.account.account_type if user.account else None,
                'balance': user.account.balance if user.account else None,
                'created_at': user.created_at
            })

        return pd.DataFrame(data)

    @staticmethod
    def get_loan_statistics():
        """
        Get loan statistics

        Returns:
            dict: Dictionary containing loan statistics
        """
        df = DataProcessor.export_loans_to_dataframe()

        if df.empty:
            return {
                'total_loans': 0,
                'total_amount': 0,
                'average_amount': 0,
                'loans_by_type': {},
                'loans_by_status': {},
                'average_interest_rate': 0
            }

        stats = {
            'total_loans': len(df),
            'total_amount': df['amount'].sum(),
            'average_amount': df['amount'].mean(),
            'loans_by_type': df['loan_type'].value_counts().to_dict(),
            'loans_by_status': df['status'].value_counts().to_dict(),
            'average_interest_rate': df['interest_rate'].mean()
        }

        return stats

    @staticmethod
    def get_user_statistics():
        """
        Get user statistics

        Returns:
            dict: Dictionary containing user statistics
        """
        df = DataProcessor.export_users_to_dataframe()

        if df.empty:
            return {
                'total_users': 0,
                'users_with_account': 0,
                'users_by_account_type': {},
                'average_balance': 0,
                'total_balance': 0
            }

        stats = {
            'total_users': len(df),
            'users_with_account': df['has_account'].sum(),
            'users_by_account_type': df[
                'account_type'].value_counts().to_dict() if 'account_type' in df.columns else {},
            'average_balance': df['balance'].mean() if 'balance' in df.columns else 0,
            'total_balance': df['balance'].sum() if 'balance' in df.columns else 0
        }

        return stats

    @staticmethod
    def analyze_loan_approval_rate():
        """
        Analyze loan approval rate by type

        Returns:
            pd.DataFrame: DataFrame containing approval rates
        """
        df = DataProcessor.export_loans_to_dataframe()

        if df.empty:
            return pd.DataFrame()

        approval_rate = df.groupby('loan_type').agg({
            'status': lambda x: (x == 'approved').sum() / len(x) * 100
        }).rename(columns={'status': 'approval_rate'})

        return approval_rate

    @staticmethod
    def get_high_value_loans(threshold=100000):
        """
        Get loans above a certain threshold

        Args:
            threshold: Minimum loan amount

        Returns:
            pd.DataFrame: DataFrame containing high value loans
        """
        df = DataProcessor.export_loans_to_dataframe()

        if df.empty:
            return pd.DataFrame()

        return df[df['amount'] >= threshold]

    @staticmethod
    def transform_loan_data():
        """
        Transform and prepare loan data for analysis

        Returns:
            pd.DataFrame: Transformed DataFrame
        """
        df = DataProcessor.export_loans_to_dataframe()

        if df.empty:
            return df

        # Convert dates to datetime
        df['application_date'] = pd.to_datetime(df['application_date'])
        df['approval_date'] = pd.to_datetime(df['approval_date'])

        # Calculate processing time
        df['processing_days'] = (df['approval_date'] - df['application_date']).dt.days

        # Calculate monthly EMI (approximate)
        df['monthly_emi'] = df.apply(
            lambda row: DataProcessor.calculate_emi(
                row['amount'],
                row['interest_rate'],
                row['tenure_months']
            ),
            axis=1
        )

        # Add risk category based on amount and interest rate
        df['risk_category'] = df.apply(
            lambda row: 'High' if row['amount'] > 500000 and row['interest_rate'] > 12
            else 'Medium' if row['amount'] > 200000 or row['interest_rate'] > 10
            else 'Low',
            axis=1
        )

        return df

    @staticmethod
    def calculate_emi(principal, annual_rate, tenure_months):
        """
        Calculate EMI (Equated Monthly Installment)

        Args:
            principal: Loan amount
            annual_rate: Annual interest rate
            tenure_months: Loan tenure in months

        Returns:
            float: Monthly EMI amount
        """
        if tenure_months == 0:
            return 0

        monthly_rate = annual_rate / (12 * 100)

        if monthly_rate == 0:
            return principal / tenure_months

        emi = (principal * monthly_rate * (1 + monthly_rate) ** tenure_months) / \
              ((1 + monthly_rate) ** tenure_months - 1)

        return round(emi, 2)