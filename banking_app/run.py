"""
Application runner with database initialization
"""
from app import create_app, db
from utils.seed_data import seed_all


def initialize_app():
    """Initialize application and database"""
    app = create_app()

    with app.app_context():
        # Create all database tables
        db.create_all()
        print("✓ Database tables created")

        # Seed loan eligibility data
        try:
            seed_all()
            print("✓ Database seeded with loan eligibility data")
        except Exception as e:
            print(f"Note: {e}")

        print("\n" + "=" * 60)
        print("Loan Approval System is ready!")
        print("=" * 60)
        print(f"API Documentation: http://localhost:5000/api/docs/")
        print(f"Server running on: http://localhost:5000")
        print("=" * 60 + "\n")

    return app


if __name__ == '__main__':
    app = initialize_app()
    app.run(debug=True, host='0.0.0.0', port=5000)