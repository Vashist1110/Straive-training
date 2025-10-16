from flask import Flask, request, jsonify
from flasgger import Swagger, swag_from
from models import BankModel

app = Flask(__name__)
app.config['SWAGGER'] = {
    'title': 'Simple Bank API',
    'uiversion': 3
}
swagger = Swagger(app)

model = BankModel()

@app.route('/account', methods=['POST'])
@swag_from({
    'tags': ['Account'],
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'schema': {
                'type': 'object',
                'properties': {
                    'tan_number': {'type': 'string'},
                    'account_holder_name': {'type': 'string'},
                    'account_type': {'type': 'string', 'enum': ['savings', 'current']}
                },
                'required': ['tan_number', 'account_holder_name', 'account_type']
            }
        }
    ],
    'responses': {
        200: {'description': 'Account creation result'},
        400: {'description': 'Invalid input'}
    }
})
def create_account():
    data = request.get_json()
    tan = data.get('tan_number', '').strip()
    name = data.get('account_holder_name', '').strip()
    acc_type = data.get('account_type')

    if not tan or not name or acc_type not in ['savings', 'current']:
        return jsonify({'error': 'Missing or invalid parameters'}), 400

    success, message = model.create_account(tan, name, acc_type)
    status = 200 if success else 400
    return jsonify({'message': message}), status


@app.route('/loan', methods=['POST'])
@swag_from({
    'tags': ['Loan'],
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'schema': {
                'type': 'object',
                'properties': {
                    'tan_number': {'type': 'string'},
                    'loan_type': {'type': 'string', 'enum': ['business_loan', 'home_loan', 'car_loan', 'education_loan', 'personal_loan']},
                    'amount': {'type': 'number', 'format': 'float'},
                },
                'required': ['loan_type', 'amount']
            }
        }
    ],
    'responses': {
        200: {'description': 'Loan application result'},
        400: {'description': 'Invalid input or loan rule violation'}
    }
})
def apply_loan():
    data = request.get_json()
    loan_type = data.get('loan_type')
    amount = data.get('amount')
    tan = data.get('tan_number', '').strip() if 'tan_number' in data else None

    if not loan_type or amount is None:
        return jsonify({'error': 'Missing loan_type or amount'}), 400

    try:
        amount_val = float(amount)
        if amount_val <= 0:
            raise ValueError()
    except:
        return jsonify({'error': 'Amount must be a positive number'}), 400

    if loan_type == 'business_loan':
        # TAN is mandatory
        if not tan:
            return jsonify({'error': 'TAN number is required for business loan'}), 400
        account = model.get_account(tan)
        if not account:
            return jsonify({'error': 'Invalid TAN number'}), 400
        if account[2] != 'current':
            return jsonify({'error': 'Business loans are only for current account holders'}), 400
    else:
        pass

    try:
        # Store loan with or without TAN
        # If no TAN, store NULL or empty string?
        tan_to_store = tan if tan else None
        model.add_loan(tan_to_store, loan_type, amount_val)
        return jsonify({'message': 'Loan application successful'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/account/<tan_number>', methods=['GET'])
@swag_from({
    'tags': ['Account'],
    'parameters': [
        {
            'name': 'tan_number',
            'in': 'path',
            'type': 'string',
            'required': True,
            'description': 'TAN number of the account'
        }
    ],
    'responses': {
        200: {
            'description': 'Account info',
            'schema': {
                'type': 'object',
                'properties': {
                    'tan_number': {'type': 'string'},
                    'account_holder_name': {'type': 'string'},
                    'account_type': {'type': 'string'}
                }
            }
        },
        404: {'description': 'Account not found'}
    }
})
def get_account(tan_number):
    account = model.get_account(tan_number)
    if not account:
        return jsonify({'error': 'Account not found'}), 404
    return jsonify({
        'tan_number': account[0],
        'account_holder_name': account[1],
        'account_type': account[2]
    })


if __name__ == '__main__':
    app.run(debug=True)
