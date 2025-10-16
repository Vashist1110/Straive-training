from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from models.user import User
from models.loan import Loan, LoanEligibility
from app import db
from services.loan_service import LoanService
from utils.exceptions import ValidationError, NotFoundError, BusinessLogicError

loan_bp = Blueprint('loan', __name__)
loan_service = LoanService()


@loan_bp.route('/apply', methods=['POST'])
@jwt_required()
def apply_loan():
    """
    Apply for a loan
    ---
    tags:
      - Loan
    security:
      - Bearer: []
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - loan_type
            - amount
            - tenure_months
          properties:
            loan_type:
              type: string
              enum: [education, business, personal, car, home]
              example: "education"
            amount:
              type: number
              example: 100000
            tenure_months:
              type: integer
              example: 60
    responses:
      201:
        description: Loan application submitted successfully
        schema:
          type: object
          properties:
            message:
              type: string
            loan:
              type: object
      400:
        description: Validation error or business logic error
      401:
        description: Unauthorized
      404:
        description: User or account not found
    """
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        required_fields = ['loan_type', 'amount', 'tenure_months']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        user = User.query.filter_by(user_id=current_user_id).first()

        if not user:
            return jsonify({'error': 'User not found'}), 404

        if not user.account:
            return jsonify({'error': 'No account found. Please create an account first'}), 400

        loan = loan_service.apply_for_loan(
            user=user,
            loan_type=data['loan_type'],
            amount=float(data['amount']),
            tenure_months=int(data['tenure_months'])
        )

        return jsonify({
            'message': 'Loan application submitted successfully',
            'loan': loan.to_dict()
        }), 201

    except ValidationError as e:
        return jsonify({'error': str(e)}), 400
    except BusinessLogicError as e:
        return jsonify({'error': str(e)}), 400
    except NotFoundError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500


@loan_bp.route('/my-loans', methods=['GET'])
@jwt_required()
def get_my_loans():
    """
    Get all loans for current user
    ---
    tags:
      - Loan
    security:
      - Bearer: []
    responses:
      200:
        description: Loans retrieved successfully
        schema:
          type: object
          properties:
            loans:
              type: array
              items:
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

        loans = [loan.to_dict() for loan in user.loans.all()]

        return jsonify({'loans': loans}), 200

    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500


@loan_bp.route('/eligibility/<loan_type>', methods=['GET'])
@jwt_required()
def check_eligibility(loan_type):
    """
    Check eligibility for a loan type
    ---
    tags:
      - Loan
    security:
      - Bearer: []
    parameters:
      - in: path
        name: loan_type
        type: string
        required: true
        enum: [education, business, personal, car, home]
        description: Type of loan
    responses:
      200:
        description: Eligibility check result
        schema:
          type: object
          properties:
            eligible:
              type: boolean
            loan_eligibility:
              type: object
            user_account:
              type: object
            reasons:
              type: array
              items:
                type: string
      400:
        description: Invalid loan type
      401:
        description: Unauthorized
      404:
        description: User or eligibility criteria not found
    """
    try:
        current_user_id = get_jwt_identity()
        user = User.query.filter_by(user_id=current_user_id).first()

        if not user:
            return jsonify({'error': 'User not found'}), 404

        if not user.account:
            return jsonify({
                'eligible': False,
                'reasons': ['No account found. Please create an account first']
            }), 200

        eligibility_result = loan_service.check_eligibility(user, loan_type)

        return jsonify(eligibility_result), 200

    except ValidationError as e:
        return jsonify({'error': str(e)}), 400
    except NotFoundError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500


@loan_bp.route('/types', methods=['GET'])
def get_loan_types():
    """
    Get all available loan types and their eligibility criteria
    ---
    tags:
      - Loan
    responses:
      200:
        description: Loan types retrieved successfully
        schema:
          type: object
          properties:
            loan_types:
              type: array
              items:
                type: object
    """
    try:
        loan_types = LoanEligibility.query.all()

        return jsonify({
            'loan_types': [loan_type.to_dict() for loan_type in loan_types]
        }), 200

    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500


@loan_bp.route('/<int:loan_id>', methods=['GET'])
@jwt_required()
def get_loan_details(loan_id):
    """
    Get loan details by ID
    ---
    tags:
      - Loan
    security:
      - Bearer: []
    parameters:
      - in: path
        name: loan_id
        type: integer
        required: true
        description: Loan ID
    responses:
      200:
        description: Loan details retrieved successfully
        schema:
          type: object
          properties:
            loan:
              type: object
      401:
        description: Unauthorized
      404:
        description: Loan not found
    """
    try:
        current_user_id = get_jwt_identity()
        user = User.query.filter_by(user_id=current_user_id).first()

        if not user:
            return jsonify({'error': 'User not found'}), 404

        loan = Loan.query.filter_by(id=loan_id, user_id=user.id).first()

        if not loan:
            return jsonify({'error': 'Loan not found'}), 404

        return jsonify({'loan': loan.to_dict()}), 200

    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500