import streamlit as st
import sqlite3
import os
from src.tools.model_evaluator import evaluate_submission

DB_PATH = 'gradebook.db'

st.set_page_config(page_title='Agent Evaluation Monitor', layout='wide')

st.title('Agent Evaluation Monitor')

if not os.path.exists(DB_PATH):
    st.error(f"Database not found at {DB_PATH}. Run scripts/init_db.py first.")
    st.stop()

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("SELECT id, nam_tuyensinh, ptxt, diem_trungtuyen FROM students LIMIT 100")
rows = cur.fetchall()

st.subheader('Students (sample)')
for r in rows:
    st.write({'id': r[0], 'year': r[1], 'ptxt': r[2], 'score': r[3]})

st.sidebar.header('Evaluate Submission')
student_id = st.sidebar.text_input('Student ID')
submission_text = st.sidebar.text_area('Submission Text (paste)', height=200)
rubric_json = st.sidebar.text_area('Rubric (JSON)', value='{"technical":40, "debugging":30, "insights":20, "future":10}', height=120)

if st.sidebar.button('Run Evaluation'):
    if not submission_text.strip():
        st.sidebar.error('Provide submission text or a submission_path in DB')
    else:
        try:
            import json
            rubric = json.loads(rubric_json)
        except Exception as e:
            st.sidebar.error(f'Invalid rubric JSON: {e}')
            rubric = {}

        with st.spinner('Calling evaluator...'):
            api_key = os.getenv('OPENAI_API_KEY')
            res = evaluate_submission(submission_text, rubric, api_key=api_key, model='gpt-4o-mini')
            st.sidebar.write(res)

st.sidebar.markdown('---')
st.sidebar.write('Notes: evaluator requires OPENAI_API_KEY in environment. If missing, call will fail.')
