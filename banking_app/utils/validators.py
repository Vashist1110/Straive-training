import re
from config import Config


def validate_pan(pan_number):
    """
    Validate PAN number format
    Format: ABCDE1234F (5 letters, 4 digits, 1 letter)

    Args:
        pan_number: PAN number string

    Returns:
        bool: True if valid, False otherwise
    """
    if not pan_number:
        return False

    pattern = r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$'
    return bool(re.match(pattern, pan_number.upper()))


def validate_tan(tan_number):
    """
    Validate TAN number format
    Format: ABCD12345E (4 letters, 5 digits, 1 letter)

    Args:
        tan_number: TAN number string

    Returns:
        bool: True if valid, False otherwise
    """
    if not tan_number:
        return False

    pattern = r'^[A-Z]{4}[0-9]{5}[A-Z]{1}$'
    return bool(re.match(pattern, tan_number.upper()))


def validate_account_type(account_type):
    """
    Validate account type

    Args:
        account_type: Account type string

    Returns:
        bool: True if valid, False otherwise
    """
    if not account_type:
        return False

    return account_type.lower() in Config.ACCOUNT_TYPES


def validate_loan_type(loan_type):
    """
    Validate loan type

    Args:
        loan_type: Loan type string

    Returns:
        bool: True if valid, False otherwise
    """
    if not loan_type:
        return False

    return loan_type.lower() in Config.LOAN_TYPES


def validate_email(email):
    """
    Validate email format

    Args:
        email: Email string

    Returns:
        bool: True if valid, False otherwise
    """
    if not email:
        return False

    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_phone(phone):
    """
    Validate phone number format (10 digits)

    Args:
        phone: Phone number string

    Returns:
        bool: True if valid, False otherwise
    """
    if not phone:
        return False

    pattern = r'^[0-9]{10}$'
    return bool(re.match(pattern, phone))