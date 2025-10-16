import pandas as pd
import sqlite3

class BankModel:
    def __init__(self, csv_path='accounts.csv', db_path='bank.db'):
        self.csv_path = csv_path
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._create_tables()
        self._load_accounts_from_csv()

    def _create_tables(self):
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS accounts (
                    tan_number TEXT PRIMARY KEY,
                    account_holder_name TEXT NOT NULL,
                    account_type TEXT CHECK(account_type IN ('savings','current')) NOT NULL
                )
            ''')
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS loans (
                    loan_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tan_number TEXT NOT NULL,
                    loan_type TEXT NOT NULL,
                    amount REAL NOT NULL,
                    FOREIGN KEY(tan_number) REFERENCES accounts(tan_number)
                )
            ''')

    def _load_accounts_from_csv(self):
        df = pd.read_csv(self.csv_path)
        with self.conn:
            for _, row in df.iterrows():
                try:
                    self.conn.execute('INSERT OR IGNORE INTO accounts VALUES (?, ?, ?)',
                                      (row['tan_number'], row['account_holder_name'], row['account_type']))
                except Exception as e:
                    print(f"Failed to insert row {row['tan_number']}: {e}")

    def get_account(self, tan_number):
        cur = self.conn.cursor()
        cur.execute('SELECT * FROM accounts WHERE tan_number = ?', (tan_number,))
        return cur.fetchone()

    def create_account(self, tan_number, name, account_type):
        with self.conn:
            try:
                self.conn.execute('INSERT INTO accounts VALUES (?, ?, ?)', (tan_number, name, account_type))
                return True, "Account created successfully"
            except sqlite3.IntegrityError as e:
                return False, f"Account creation failed: {e}"

    def add_loan(self, tan_number, loan_type, amount):
        with self.conn:
            self.conn.execute('INSERT INTO loans (tan_number, loan_type, amount) VALUES (?, ?, ?)',
                              (tan_number, loan_type, amount))
