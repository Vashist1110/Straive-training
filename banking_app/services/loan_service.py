from models.loan import Loan, LoanEligibility
from models.user import User
from app import db
from utils.exceptions import ValidationError, NotFoundError, BusinessLogicError
from config import Config
from datetime import datetime
import random
import string


class LoanService:
    """Service class for loan operations"""

    def generate_loan_number(self):
        """Generate unique loan number"""
        while True:
            loan_number = 'LN' + ''.join(random.choices(string.digits, k=10))
            existing = Loan.query.filter_by(loan_number=loan_number).first()
            if not existing:
                return loan_number

    def calculate_interest_rate(self, loan_type, amount, tenure_months, user):
        """
        Calculate interest rate based on loan type and user profile

        Args:
            loan_type: Type of loan
            amount: Loan amount
            tenure_months: Loan tenure in months
            user: User object

        Returns:
            float: Interest rate
        """
        eligibility = LoanEligibility.query.filter_by(loan_type=loan_type).first()

        if not eligibility:
            return 10.0  # Default rate

        # Base rate
        base_rate = eligibility.min_interest_rate

        # Adjust based on amount
        if amount > eligibility.max_amount * 0.8:
            base_rate += 1.0

        # Adjust based on tenure
        if tenure_months > eligibility.max_tenure_months * 0.8:
            base_rate += 0.5

        # Adjust based on account balance
        if user.account and user.account.balance > amount * 0.5:
            base_rate -= 0.5

        # Ensure rate is within bounds
        return min(max(base_rate, eligibility.min_interest_rate), eligibility.max_interest_rate)

    def check_eligibility(self, user, loan_type):
        """
        Check loan eligibility for user

        Args:
            user: User object
            loan_type: Type of loan

        Returns:
            dict: Eligibility result with reasons

        Raises:
            ValidationError: If loan type is invalid
            NotFoundError: If eligibility criteria not found
        """
        if loan_type not in Config.LOAN_TYPES:
            raise ValidationError(f'Invalid loan type. Must be one of: {", ".join(Config.LOAN_TYPES)}')

        eligibility = LoanEligibility.query.filter_by(loan_type=loan_type).first()

        if not eligibility:
            raise NotFoundError(f'Eligibility criteria not found for loan type: {loan_type}')

        reasons = []
        eligible = True

        # Check if user has account
        if not user.account:
            eligible = False
            reasons.append('No account found. Please create an account first')
            return {
                'eligible': eligible,
                'loan_eligibility': eligibility.to_dict(),
                'reasons': reasons
            }

        # Check account type for business loan
        if eligibility.requires_current_account and user.account.account_type != 'current':
            eligible = False
            reasons.append('Business loan requires a current account')

        # Check TAN number for business loan
        if loan_type == 'business' and not user.tan_number:
            eligible = False
            reasons.append('TAN number is required for business loan')

        # Check minimum balance
        if user.account.balance < eligibility.min_balance:
            eligible = False
            reasons.append(f'Insufficient balance. Minimum balance of {eligibility.min_balance} required')

        # Check if account is active
        if not user.account.is_active:
            eligible = False
            reasons.append('Account is not active')

        if eligible:
            reasons.append('You are eligible for this loan')

        return {
            'eligible': eligible,
            'loan_eligibility': eligibility.to_dict(),
            'user_account': user.account.to_dict(),
            'reasons': reasons
        }

    def apply_for_loan(self, user, loan_type, amount, tenure_months):
        """
        Apply for a loan

        Args:
            user: User object
            loan_type: Type of loan
            amount: Loan amount
            tenure_months: Loan tenure in months

        Returns:
            Loan: Loan object

        Raises:
            ValidationError: If validation fails
            BusinessLogicError: If business rules are violated
        """
        # Validate loan type
        if loan_type not in Config.LOAN_TYPES:
            raise ValidationError(f'Invalid loan type. Must be one of: {", ".join(Config.LOAN_TYPES)}')

        # Get eligibility criteria
        eligibility = LoanEligibility.query.filter_by(loan_type=loan_type).first()

        if not eligibility:
            raise NotFoundError(f'Eligibility criteria not found for loan type: {loan_type}')

        # Check eligibility
        eligibility_result = self.check_eligibility(user, loan_type)

        if not eligibility_result['eligible']:
            raise BusinessLogicError(
                f"Not eligible for {loan_type} loan. Reasons: {', '.join(eligibility_result['reasons'])}"
            )

        # Validate amount
        if amount < 0:
            raise ValidationError('Loan amount must be greater than 0')

        if amount > eligibility.max_amount:
            raise ValidationError(f'Loan amount exceeds maximum limit of {eligibility.max_amount}')

        # Validate tenure
        if tenure_months < eligibility.min_tenure_months:
            raise ValidationError(
                f'Loan tenure must be at least {eligibility.min_tenure_months} months'
            )

        if tenure_months > eligibility.max_tenure_months:
            raise ValidationError(
                f'Loan tenure cannot exceed {eligibility.max_tenure_months} months'
            )

        # Calculate interest rate
        interest_rate = self.calculate_interest_rate(loan_type, amount, tenure_months, user)

        # Create loan
        loan = Loan(
            loan_number=self.generate_loan_number(),
            user_id=user.id,
            loan_type=loan_type,
            amount=amount,
            interest_rate=interest_rate,
            tenure_months=tenure_months,
            status='pending'
        )

        db.session.add(loan)
        db.session.commit()

        return loan

    def approve_loan(self, loan):
        """
        Approve a loan

        Args:
            loan: Loan object

        Returns:
            Loan: Updated loan object
        """
        if loan.status != 'pending':
            raise BusinessLogicError('Only pending loans can be approved')

        loan.status = 'approved'
        loan.approval_date = datetime.utcnow()

        return loan

    def reject_loan(self, loan, reason):
        """
        Reject a loan

        Args:
            loan: Loan object
            reason: Rejection reason

        Returns:
            Loan: Updated loan object
        """
        if loan.status != 'pending':
            raise BusinessLogicError('Only pending loans can be rejected')

        loan.status = 'rejected'
        loan.rejection_reason = reason

        return loan