import pandas as pd
import sqlite3

df = pd.read_csv("employee.csv")

print("Original Data:")
print(df)


df.loc[df['department'] == 'IT', 'salary'] *= 1.10


df['name'] = df['name'].str.upper()


df['bonus'] = df['salary'] * 0.05

print("\nTransformed Data:")
print(df)


conn = sqlite3.connect("employees.db")


df.to_sql("employees", conn, if_exists="replace", index=False)

print("\nData has been loaded into employees.db (table: employees)")


result = pd.read_sql("SELECT * FROM employees", conn)

print("\nData from DB:")
print(result)


conn.close()