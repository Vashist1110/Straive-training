#!/bin/bash

# --- Environment Setup ---
echo "--- Setting up Python environment and dependencies ---"
# Check if pip is installed
if ! command -v pip &> /dev/null
then
    echo "pip not found. Please ensure Python and pip are installed."
    exit 1
fi

# Install necessary Python packages
pip install flask flasgger pandas
if [ $? -ne 0 ]; then
    echo "Failed to install dependencies (Flask, Flasgger, Pandas)."
    exit 1
fi

# --- DB/Data Initialization ---
DB_FILE="bank_data.json"
if [ ! -f "$DB_FILE" ]; then
    echo "--- Initializing bank_data.json (simulated database) ---"
    python -c "import json; f=open('$DB_FILE', 'w'); json.dump({'accounts': [], 'loan_rules': {}}, f, indent=2); f.close();"
fi

# --- Run Unit Tests ---
echo "--- Running Unit Tests (tests.py) ---"
python tests.py
echo "--- Unit Tests Complete ---"

# --- Run Flask Application ---
echo "--- Starting Flask Application (app.py) ---"
# Set environment variable for Flask
export FLASK_APP=app.py
# Run the application (will run on http://127.0.0.1:5000 by default)
echo "API running at http://127.0.0.1:5000"
echo "Swagger UI available at http://127.0.0.1:5000/apidocs"
flask run
