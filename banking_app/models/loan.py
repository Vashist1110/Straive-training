from app import db
from datetime import datetime


class Loan(db.Model):
    __tablename__ = 'loans'

    id = db.Column(db.Integer, primary_key=True)
    loan_number = db.Column(db.String(20), unique=True, nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    loan_type = db.Column(db.String(20), nullable=False)  # education, business, personal, car, home
    amount = db.Column(db.Float, nullable=False)
    interest_rate = db.Column(db.Float, nullable=False)
    tenure_months = db.Column(db.Integer, nullable=False)
    status = db.Column(db.String(20), default='pending')  # pending, approved, rejected, disbursed, closed
    application_date = db.Column(db.DateTime, default=datetime.utcnow)
    approval_date = db.Column(db.DateTime, nullable=True)
    rejection_reason = db.Column(db.String(500), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'loan_number': self.loan_number,
            'loan_type': self.loan_type,
            'amount': self.amount,
            'interest_rate': self.interest_rate,
            'tenure_months': self.tenure_months,
            'status': self.status,
            'application_date': self.application_date.isoformat() if self.application_date else None,
            'approval_date': self.approval_date.isoformat() if self.approval_date else None,
            'rejection_reason': self.rejection_reason,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

    def __repr__(self):
        return f'<Loan {self.loan_number} - {self.loan_type}>'


class LoanEligibility(db.Model):
    __tablename__ = 'loan_eligibility'

    id = db.Column(db.Integer, primary_key=True)
    loan_type = db.Column(db.String(20), unique=True, nullable=False)
    min_balance = db.Column(db.Float, nullable=False)
    max_amount = db.Column(db.Float, nullable=False)
    min_interest_rate = db.Column(db.Float, nullable=False)
    max_interest_rate = db.Column(db.Float, nullable=False)
    min_tenure_months = db.Column(db.Integer, nullable=False)
    max_tenure_months = db.Column(db.Integer, nullable=False)
    requires_current_account = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'loan_type': self.loan_type,
            'min_balance': self.min_balance,
            'max_amount': self.max_amount,
            'min_interest_rate': self.min_interest_rate,
            'max_interest_rate': self.max_interest_rate,
            'min_tenure_months': self.min_tenure_months,
            'max_tenure_months': self.max_tenure_months,
            'requires_current_account': self.requires_current_account
        }

    def __repr__(self):
        return f'<LoanEligibility {self.loan_type}>'