from flask import Flask, jsonify, request
import pandas as pd
import sqlite3

app = Flask(__name__)
DB_FILE = "student.db"
TABLE_NAME = "students"

class ETLError(Exception):
    pass

def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

@app.route("/students",methods=["GET"])
def get_all_students():
    try:
        conn = get_db_connection()
        df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
        conn.close()
        return jsonify(df.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}),500

@app.route("/students/<int:roll_no>", methods=["GET"])
def get_student_by_roll(roll_no):
    try:
        conn = get_db_connection()
        df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME} WHERE roll_no = ?", conn, params=(roll_no,))
        conn.close()
        if df.empty:
            return jsonify({"message": "Student not found"}), 404
        return jsonify(df.to_dict(orient="records")[0])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/students/passed", methods=["GET"])
def get_passed_students():
    try:
        conn = get_db_connection()
        df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME} WHERE cgpa >= 6", conn)
        conn.close()
        return jsonify(df.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/students/failed", methods=["GET"])
def get_failed_students():
    try:
        conn = get_db_connection()
        df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME} WHERE cgpa < 6", conn)
        conn.close()
        return jsonify(df.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/students", methods=["POST"])
def add_student():
    try:
        data = request.json

        required_fields = {"name", "degree", "cgpa"}
        if not data or not required_fields.issubset(data.keys()):
            return jsonify({"error": f"Required fields missing. Needed: {required_fields}"}), 400


        try:
            cgpa = float(data['cgpa'])
            if cgpa < 0:
                return jsonify({"error": "CGPA cannot be negative"}), 400
        except ValueError:
            return jsonify({"error": "CGPA must be a number"}), 400

        conn = get_db_connection()
        max_roll = conn.execute(f"SELECT MAX(roll_no) FROM {TABLE_NAME}").fetchone()[0]
        new_roll_no = (max_roll or 0) + 1

        conn.execute(f"""
            INSERT INTO {TABLE_NAME} (roll_no, name, degree, cgpa) 
            VALUES (?, ?, ?, ?)
        """, (new_roll_no, data['name'], data['degree'], cgpa))
        conn.commit()
        conn.close()
        return jsonify({"message": "Student added", "roll_no": new_roll_no}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/students/<int:roll_no>", methods=["DELETE"])
def delete_student(roll_no):
    try:
        conn = get_db_connection()
        cur = conn.execute(f"DELETE FROM {TABLE_NAME} WHERE roll_no = ?", (roll_no,))
        conn.commit()
        conn.close()
        if cur.rowcount == 0:
            return jsonify({"message": "Student not found"}), 404
        return jsonify({"message": f"Student with roll_no {roll_no} deleted"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)