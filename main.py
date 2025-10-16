from bank import BankSystem
from accounts import SavingsAccount


def display_menu():
    print("\nBANKING SYSTEM MENU")
    print("1.Create New Account")
    print("2.Deposit Money")
    print("3.Withdraw Money")
    print("4.Transfer Money")
    print("5.View Account Details")
    print("6.View Transaction History")
    print("7.Calculate Interest (Savings Only)")
    print("8.Close Account")
    print("9.List All Accounts")
    print("10Manager Operations")
    print("11Exit")


def manager_menu(bank_system):
    print("\nMANAGER MENU")
    print("1.Add Manager")
    print("2.Approve Loan")
    print("3.View Approved Loans")
    print("4.Back to Main Menu")

    choice = input("Enter choice: ")

    if choice == '1':
        emp_id = input("Enter Manager ID: ")
        name = input("Enter Manager Name: ")
        manager = bank_system.add_manager(emp_id, name)
        print(f"Manager {name} added successfully.")

    elif choice == '2':
        managers = bank_system.get_managers()
        if not managers:
            print("No managers found.")
            return

        print("\nAvailable Managers:")
        for idx, mgr in enumerate(managers, 1):
            print(f"{idx}. {mgr.name}")

        mgr_index = int(input("Choose Manager: ")) - 1
        if 0 <= mgr_index < len(managers):
            acc_num = input("Enter Account Number: ")
            account = bank_system.find_account(acc_num)
            if account:
                amount = float(input("Enter Loan Amount: Rs"))
                rate = float(input("Enter Interest Rate (%): "))
                success, msg = managers[mgr_index].approve_loan(account, amount, rate)
                print(msg)
            else:
                print("Account not found.")

    elif choice == '3':
        managers = bank_system.get_managers()
        if not managers:
            print("No managers found.")
            return

        print("\nAvailable Managers:")
        for idx, mgr in enumerate(managers, 1):
            print(f"{idx}. {mgr.name}")

        mgr_index = int(input("Choose Manager: ")) - 1
        if 0 <= mgr_index < len(managers):
            managers[mgr_index].view_approved_loans()


def main():
    bank = BankSystem()
    print("Welcome to the Modular Banking System!")

    while True:
        display_menu()
        choice = input("Enter your choice: ")

        try:
            if choice == '1':
                holder = input("Enter account holder name: ")
                acc_type = input("Enter account type (savings/current): ")
                balance = float(input("Enter initial balance: Rs"))

                if acc_type.lower() == 'current':
                    # overdraft = float(input("Enter overdraft limit: Rs"))
                    account, msg = bank.create_account(acc_type, holder, balance)
                else:
                    account, msg = bank.create_account(acc_type, holder, balance)

                print(msg)
                if account:
                    print(f"Account Number: {account.account_number}")

            elif choice == '2':
                acc_num = input("Enter account number: ")
                acc = bank.find_account(acc_num)
                if acc:
                    amount = float(input("Enter amount to deposit: RsRs."))
                    success, msg = acc.deposit(amount)
                    print(msg)
                else:
                    print("Account not found.")

            elif choice == '3':
                acc_num = input("Enter account number: ")
                acc = bank.find_account(acc_num)
                if acc:
                    amount = float(input("Enter amount to withdraw: Rs"))
                    success, msg = acc.withdraw(amount)
                    print(msg)
                else:
                    print("Account not found.")

            elif choice == '4':
                src = input("Enter source account number: ")
                tgt = input("Enter target account number: ")
                src_acc = bank.find_account(src)
                tgt_acc = bank.find_account(tgt)
                if src_acc and tgt_acc:
                    amount = float(input("Enter amount to transfer: Rs."))
                    success, msg = src_acc.transfer(amount, tgt_acc)
                    print(msg)
                else:
                    print("One or both accounts not found.")

            elif choice == '5':
                acc_num = input("Enter account number: ")
                acc = bank.find_account(acc_num)
                if acc:
                    acc.display_details()
                else:
                    print("Account not found.")

            elif choice == '6':
                acc_num = input("Enter account number: ")
                acc = bank.find_account(acc_num)
                if acc:
                    acc.display_transaction_history()
                else:
                    print("Account not found.")

            elif choice == '7':
                acc_num = input("Enter account number: ")
                acc = bank.find_account(acc_num)
                if isinstance(acc, SavingsAccount):
                    success, msg = acc.calculate_interest()
                    print(msg)
                else:
                    print("Only Savings accounts support interest.")

            elif choice == '8':
                acc_num = input("Enter account number: ")
                acc = bank.find_account(acc_num)
                if acc:
                    success, msg = acc.close_account()
                    print(msg)
                else:
                    print("Account not found.")

            elif choice == '9':
                accounts = bank.get_all_accounts()
                if accounts:
                    print(f"\n{'=' * 60}")
                    print(f"{'Acc Number':<15} {'Holder':<20} {'Type':<10} {'Balance':>10}")
                    print(f"{'=' * 60}")
                    for acc in accounts:
                        print(f"{acc.account_number:<15} {acc.account_holder:<20} "
                              f"{acc.__class__.__name__:<10} Rs.{acc.balance:>9.2f}")
                    print(f"{'=' * 60}")
                else:
                    print("No accounts found.")

            elif choice == '10':
                manager_menu(bank)

            elif choice == '11':
                print("Thank you for using the banking system!")
                break

            else:
                print("Invalid choice. Try again.")

        except ValueError as ve:
            print(f"Invalid input: {ve}")
        except Exception as e:
            print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
