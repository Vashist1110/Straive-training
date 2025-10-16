class ValidationError(Exception):
    """Exception raised for validation errors"""
    pass


class DatabaseError(Exception):
    """Exception raised for database errors"""
    pass


class NotFoundError(Exception):
    """Exception raised when resource is not found"""
    pass


class BusinessLogicError(Exception):
    """Exception raised for business logic violations"""
    pass


class UnauthorizedError(Exception):
    """Exception raised for unauthorized access"""
    pass


class AuthenticationError(Exception):
    """Exception raised for authentication failures"""
    pass