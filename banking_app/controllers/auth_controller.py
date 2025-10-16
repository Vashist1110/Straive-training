from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token, create_refresh_token, jwt_required, get_jwt_identity
from flasgger import swag_from
from models.user import User, Account
from app import db
from services.auth_service import AuthService
from utils.validators import validate_pan, validate_tan, validate_account_type
from utils.exceptions import ValidationError, DatabaseError
import random
import string

auth_bp = Blueprint('auth', __name__)
auth_service = AuthService()


@auth_bp.route('/register', methods=['POST'])
def register():
    """
    Register a new user and create account
    ---
    tags:
      - Authentication
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - name
            - pan_number
            - user_id
            - password
            - initial_amount
            - account_type
          properties:
            name:
              type: string
              example: "John Doe"
            pan_number:
              type: string
              example: "ABCDE1234F"
            tan_number:
              type: string
              example: "DELA12345E"
            user_id:
              type: string
              example: "john_doe_123"
            password:
              type: string
              example: "SecurePass@123"
            initial_amount:
              type: number
              example: 10000
            account_type:
              type: string
              enum: [savings, current]
              example: "savings"
    responses:
      201:
        description: User registered successfully
        schema:
          type: object
          properties:
            message:
              type: string
            user:
              type: object
            access_token:
              type: string
            refresh_token:
              type: string
      400:
        description: Validation error
      409:
        description: User already exists
      500:
        description: Internal server error
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Validate required fields
        required_fields = ['name', 'pan_number', 'user_id', 'password', 'initial_amount', 'account_type']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Validate PAN
        if not validate_pan(data['pan_number']):
            return jsonify({'error': 'Invalid PAN number format'}), 400

        # Validate TAN if provided
        if 'tan_number' in data and data['tan_number']:
            if not validate_tan(data['tan_number']):
                return jsonify({'error': 'Invalid TAN number format'}), 400

        # Validate account type
        if not validate_account_type(data['account_type']):
            return jsonify({'error': 'Invalid account type. Must be savings or current'}), 400

        # Register user
        user, account = auth_service.register_user(
            name=data['name'],
            pan_number=data['pan_number'].upper(),
            tan_number=data.get('tan_number', '').upper() if data.get('tan_number') else None,
            user_id=data['user_id'],
            password=data['password'],
            initial_amount=float(data['initial_amount']),
            account_type=data['account_type']
        )

        # Create tokens
        access_token = create_access_token(identity=user.user_id)
        refresh_token = create_refresh_token(identity=user.user_id)

        return jsonify({
            'message': 'User registered successfully',
            'user': user.to_dict(),
            'access_token': access_token,
            'refresh_token': refresh_token
        }), 201

    except ValidationError as e:
        return jsonify({'error': str(e)}), 400
    except DatabaseError as e:
        return jsonify({'error': str(e)}), 409
    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500


@auth_bp.route('/login', methods=['POST'])
def login():
    """
    Login user
    ---
    tags:
      - Authentication
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - user_id
            - password
          properties:
            user_id:
              type: string
              example: "john_doe_123"
            password:
              type: string
              example: "SecurePass@123"
    responses:
      200:
        description: Login successful
        schema:
          type: object
          properties:
            message:
              type: string
            user:
              type: object
            access_token:
              type: string
            refresh_token:
              type: string
      401:
        description: Invalid credentials
      404:
        description: User not found
      500:
        description: Internal server error
    """
    try:
        data = request.get_json()

        if not data or 'user_id' not in data or 'password' not in data:
            return jsonify({'error': 'User ID and password are required'}), 400

        user = auth_service.authenticate_user(data['user_id'], data['password'])

        if not user:
            return jsonify({'error': 'Invalid user ID or password'}), 401

        access_token = create_access_token(identity=user.user_id)
        refresh_token = create_refresh_token(identity=user.user_id)

        return jsonify({
            'message': 'Login successful',
            'user': user.to_dict(),
            'access_token': access_token,
            'refresh_token': refresh_token
        }), 200

    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500


@auth_bp.route('/refresh', methods=['POST'])
@jwt_required(refresh=True)
def refresh():
    """
    Refresh access token
    ---
    tags:
      - Authentication
    security:
      - Bearer: []
    responses:
      200:
        description: Token refreshed successfully
        schema:
          type: object
          properties:
            access_token:
              type: string
      401:
        description: Invalid or expired refresh token
    """
    try:
        current_user = get_jwt_identity()
        access_token = create_access_token(identity=current_user)
        return jsonify({'access_token': access_token}), 200
    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500