from app import db
from models.loan import LoanEligibility


def seed_loan_eligibility():
    """Seed loan eligibility data"""

    # Check if data already exists
    existing = LoanEligibility.query.first()
    if existing:
        print("Loan eligibility data already exists")
        return

    eligibility_data = [
        {
            'loan_type': 'education',
            'min_balance': 5000,
            'max_amount': 1000000,
            'min_interest_rate': 8.5,
            'max_interest_rate': 12.0,
            'min_tenure_months': 12,
            'max_tenure_months': 84,
            'requires_current_account': False
        },
        {
            'loan_type': 'business',
            'min_balance': 50000,
            'max_amount': 5000000,
            'min_interest_rate': 10.0,
            'max_interest_rate': 15.0,
            'min_tenure_months': 12,
            'max_tenure_months': 120,
            'requires_current_account': True
        },
        {
            'loan_type': 'personal',
            'min_balance': 10000,
            'max_amount': 500000,
            'min_interest_rate': 11.0,
            'max_interest_rate': 16.0,
            'min_tenure_months': 12,
            'max_tenure_months': 60,
            'requires_current_account': False
        },
        {
            'loan_type': 'car',
            'min_balance': 20000,
            'max_amount': 2000000,
            'min_interest_rate': 9.0,
            'max_interest_rate': 13.0,
            'min_tenure_months': 12,
            'max_tenure_months': 84,
            'requires_current_account': False
        },
        {
            'loan_type': 'home',
            'min_balance': 100000,
            'max_amount': 10000000,
            'min_interest_rate': 8.0,
            'max_interest_rate': 11.0,
            'min_tenure_months': 60,
            'max_tenure_months': 360,
            'requires_current_account': False
        }
    ]

    for data in eligibility_data:
        eligibility = LoanEligibility(**data)
        db.session.add(eligibility)

    db.session.commit()
    print("Loan eligibility data seeded successfully")


def seed_all():
    """Seed all required data"""
    seed_loan_eligibility()