import sqlite3, json
from src.agent.agent import ReActAgent
from src.core.llm_provider import LLMProvider

class FakeLLM(LLMProvider):
    def __init__(self, model_name='fake'):
        super().__init__(model_name)
        self.next_response = ''
    def generate(self, prompt, system_prompt=None):
        return {'content': self.next_response, 'usage': None, 'latency_ms': 0}
    def stream(self, prompt, system_prompt=None):
        yield self.next_response

def main():
    conn = sqlite3.connect('gradebook.db')
    cur = conn.cursor()
    cur.execute('SELECT id FROM students LIMIT 1')
    row = cur.fetchone()
    if not row:
        print('No students in DB')
        return
    student_id = row[0]
    print('Using student_id:', student_id)

    fake = FakeLLM()
    action = json.dumps({'student_id': student_id})
    fake.next_response = f"Thought: I will fetch the student info.\nAction: db_query({action})\nObservation:"

    tools = [
        {'name':'db_query','description':'Get student info by student_id'},
        {'name':'model_eval','description':'Evaluate submission'}
    ]

    agent = ReActAgent(llm=fake, tools=tools, max_steps=3)
    out = agent.run('Please return info for the student')
    print('\nAgent returned:')
    print(out)

if __name__ == '__main__':
    main()
