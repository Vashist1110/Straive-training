import pandas as pd
import sqlite3


class ETLError(Exception):
    pass


def run_etl(csv_file, db_file="student.db"):
    try:
        df = pd.read_csv(csv_file)
        if df.empty:
            raise ETLError("CSV file is empty!")

        df['cgpa'] = df['cgpa'].apply(lambda x: float(x) if x >= 0 else 0)


        conn = sqlite3.connect(db_file)
        df.to_sql("students", conn, if_exists="replace", index=False)
        df_loaded = pd.read_sql_query("SELECT * FROM students",conn)
        passed_students = pd.read_sql_query("select * from students where cgpa>=6",conn)
        failed_stu_name = pd.read_sql_query("select name from students where cgpa<6",conn)
        conn.close()

        print("ETL Completed Successfully!")
        print(df_loaded)
        print("\n")
        print("Data of passed students")
        print(passed_students)
        print("\n Name of students who failed are: ")
        print(failed_stu_name)
    except FileNotFoundError:
        print(f"Error: File {csv_file} not found.")
    except ETLError as e:
        print(f"ETL Error: {e}")
    except Exception as e:
        print(f"Unexpected Error: {e}")


if __name__ == "__main__":
    run_etl("student_data.csv")
