import pytest
import os
import sqlite3
from src.tools.data_access import DataAccess
from src.tools.db_tool import DBTool

@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "test_gradebook.db"
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute('''CREATE TABLE students (
        id TEXT PRIMARY KEY,
        nam_tuyensinh TEXT,
        ptxt TEXT,
        diem_trungtuyen REAL
    )''')
    cur.execute('''CREATE TABLE submissions (
        id INTEGER PRIMARY KEY,
        student_id TEXT,
        score REAL,
        status TEXT
    )''')
    cur.execute("INSERT INTO students VALUES ('SV001', '2023', 'THPT', 25.5)")
    cur.execute("INSERT INTO submissions (student_id, score, status) VALUES ('SV001', 9.0, 'ok')")
    conn.commit()
    conn.close()
    return str(db_file)

def test_data_access_sqlite(temp_db):
    da = DataAccess(db_path=temp_db)
    student = da.get_student("SV001")
    assert student is not None
    assert student["id"] == "SV001"
    assert student["diem_trungtuyen"] == 25.5

    submissions = da.get_submissions("SV001")
    assert len(submissions) == 1
    assert submissions[0]["score"] == 9.0

def test_data_access_sqlite_not_found(temp_db):
    da = DataAccess(db_path=temp_db)
    student = da.get_student("SV999")
    assert student is None

def test_db_tool_wrapper(temp_db):
    tool = DBTool(db_path=temp_db)
    res = tool.get_student_info("SV001")
    
    assert res is not None
    assert "student" in res
    assert res["student"]["id"] == "SV001"
    
    assert "summary" in res
    assert res["summary"]["num_submissions"] == 1
