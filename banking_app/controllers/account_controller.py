from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from models.user import User, Account
from app import db
from services.account_service import AccountService
from utils.validators import validate_account_type
from utils.exceptions import ValidationError, NotFoundError

account_bp = Blueprint('account', __name__)
account_service = AccountService()


@account_bp.route('/profile', methods=['GET'])
@jwt_required()
def get_profile():
    """
    Get user profile and account details
    ---
    tags:
      - Account
    security:
      - Bearer: []
    responses:
      200:
        description: User profile retrieved successfully
        schema:
          type: object
          properties:
            user:
              type: object
      401:
        description: Unauthorized
      404:
        description: User not found
    """
    try:
        current_user_id = get_jwt_identity()
        user = User.query.filter_by(user_id=current_user_id).first()

        if not user:
            return jsonify({'error': 'User not found'}), 404

        return jsonify({'user': user.to_dict()}), 200

    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500


@account_bp.route('/modify-type', methods=['PUT'])
@jwt_required()
def modify_account_type():
    """
    Modify account type (savings to current or vice versa)
    ---
    tags:
      - Account
    security:
      - Bearer: []
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - new_account_type
          properties:
            new_account_type:
              type: string
              enum: [savings, current]
              example: "current"
            tan_number:
              type: string
              example: "DELA12345E"
              description: Required when changing to current account
    responses:
      200:
        description: Account type modified successfully
        schema:
          type: object
          properties:
            message:
              type: string
            account:
              type: object
      400:
        description: Validation error
      401:
        description: Unauthorized
      404:
        description: Account not found
    """
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()

        if not data or 'new_account_type' not in data:
            return jsonify({'error': 'New account type is required'}), 400

        new_account_type = data['new_account_type']

        if not validate_account_type(new_account_type):
            return jsonify({'error': 'Invalid account type. Must be savings or current'}), 400

        user = User.query.filter_by(user_id=current_user_id).first()

        if not user:
            return jsonify({'error': 'User not found'}), 404

        if not user.account:
            return jsonify({'error': 'No account found for this user'}), 404

        # Check if TAN number is required for current account
        if new_account_type == 'current' and not user.tan_number:
            tan_number = data.get('tan_number')
            if not tan_number:
                return jsonify({'error': 'TAN number is required for current account'}), 400

            from utils.validators import validate_tan
            if not validate_tan(tan_number):
                return jsonify({'error': 'Invalid TAN number format'}), 400

            user.tan_number = tan_number.upper()

        account = account_service.modify_account_type(user.account, new_account_type)
        db.session.commit()

        return jsonify({
            'message': 'Account type modified successfully',
            'account': account.to_dict()
        }), 200

    except ValidationError as e:
        return jsonify({'error': str(e)}), 400
    except NotFoundError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500


@account_bp.route('/balance', methods=['GET'])
@jwt_required()
def get_balance():
    """
    Get account balance
    ---
    tags:
      - Account
    security:
      - Bearer: []
    responses:
      200:
        description: Balance retrieved successfully
        schema:
          type: object
          properties:
            account_number:
              type: string
            balance:
              type: number
            account_type:
              type: string
      401:
        description: Unauthorized
      404:
        description: Account not found
    """
    try:
        current_user_id = get_jwt_identity()
        user = User.query.filter_by(user_id=current_user_id).first()

        if not user or not user.account:
            return jsonify({'error': 'Account not found'}), 404

        return jsonify({
            'account_number': user.account.account_number,
            'balance': user.account.balance,
            'account_type': user.account.account_type
        }), 200

    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500


@account_bp.route('/deposit', methods=['POST'])
@jwt_required()
def deposit():
    """
    Deposit money to account
    ---
    tags:
      - Account
    security:
      - Bearer: []
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - amount
          properties:
            amount:
              type: number
              example: 5000
    responses:
      200:
        description: Deposit successful
        schema:
          type: object
          properties:
            message:
              type: string
            new_balance:
              type: number
      400:
        description: Invalid amount
      401:
        description: Unauthorized
      404:
        description: Account not found
    """
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()

        if not data or 'amount' not in data:
            return jsonify({'error': 'Amount is required'}), 400

        amount = float(data['amount'])

        if amount <= 0:
            return jsonify({'error': 'Amount must be greater than 0'}), 400

        user = User.query.filter_by(user_id=current_user_id).first()

        if not user or not user.account:
            return jsonify({'error': 'Account not found'}), 404

        user.account.balance += amount
        db.session.commit()

        return jsonify({
            'message': 'Deposit successful',
            'new_balance': user.account.balance
        }), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500