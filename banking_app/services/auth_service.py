from models.user import User, Account
from app import db
from utils.exceptions import ValidationError, DatabaseError
from config import Config
import random
import string


class AuthService:
    """Service class for authentication operations"""

    def generate_account_number(self):
        """Generate unique account number"""
        while True:
            account_number = ''.join(random.choices(string.digits, k=12))
            existing = Account.query.filter_by(account_number=account_number).first()
            if not existing:
                return account_number

    def register_user(self, name, pan_number, user_id, password, initial_amount, account_type, tan_number=None):
        """
        Register a new user with account

        Args:
            name: User's full name
            pan_number: PAN number
            user_id: Unique user ID
            password: User password
            initial_amount: Initial deposit amount
            account_type: Type of account (savings/current)
            tan_number: TAN number (optional, required for current account)

        Returns:
            tuple: (User, Account) objects

        Raises:
            ValidationError: If validation fails
            DatabaseError: If user already exists
        """
        try:
            # Check if user already exists
            existing_user = User.query.filter(
                (User.user_id == user_id) | (User.pan_number == pan_number)
            ).first()

            if existing_user:
                if existing_user.user_id == user_id:
                    raise DatabaseError('User ID already exists')
                if existing_user.pan_number == pan_number:
                    raise DatabaseError('PAN number already registered')

            # Validate initial amount
            min_balance = Config.MIN_BALANCE.get(account_type, 0)
            if initial_amount < min_balance:
                raise ValidationError(
                    f'Initial amount must be at least {min_balance} for {account_type} account'
                )

            # For current account, TAN is required
            if account_type == 'current' and not tan_number:
                raise ValidationError('TAN number is required for current account')

            # Check if TAN already exists
            if tan_number:
                existing_tan = User.query.filter_by(tan_number=tan_number).first()
                if existing_tan:
                    raise DatabaseError('TAN number already registered')

            # Create user
            user = User(
                user_id=user_id,
                name=name,
                pan_number=pan_number,
                tan_number=tan_number
            )
            user.set_password(password)

            # Create account
            account = Account(
                account_number=self.generate_account_number(),
                account_type=account_type,
                balance=initial_amount,
                user=user
            )

            db.session.add(user)
            db.session.add(account)
            db.session.commit()

            return user, account

        except (ValidationError, DatabaseError):
            db.session.rollback()
            raise
        except Exception as e:
            db.session.rollback()
            raise DatabaseError(f'Failed to register user: {str(e)}')

    def authenticate_user(self, user_id, password):
        """
        Authenticate user with credentials

        Args:
            user_id: User ID
            password: User password

        Returns:
            User: User object if authentication successful, None otherwise
        """
        user = User.query.filter_by(user_id=user_id).first()

        if not user:
            return None

        if not user.check_password(password):
            return None

        return user