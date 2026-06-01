# Báo cáo Giai đoạn 2: Thiết kế Tool

## 1. Các tool đã được xây dựng

Nhóm chúng tôi (đặc biệt là Dev A) đã xây dựng một bộ tools để hỗ trợ Agent tương tác với hệ thống:

1. **`db_tool`**: Hỗ trợ tra cứu thông tin học sinh (`get_student_info(student_id)`).
2. **`model_evaluator`**: Hỗ trợ gọi mô hình LLM để đánh giá văn bản dựa trên rubric (`evaluate_submission(submission_text, rubric_json)`).
3. **`DataAccess`**: Component nền tảng hỗ trợ truy xuất SQL SQLite hoặc CSV.

## 2. Sự tiến hóa của Tool Spec (v1 → v2)

Trong quá trình thiết kế, mô tả (description) của các tool đã được cải thiện để giúp LLM (ReAct Agent) hiểu và sử dụng chính xác hơn. 

### `db_query` Tool

**Version 1 (Vague)**:
```json
{
  "name": "db_query",
  "description": "Lấy thông tin học sinh."
}
```
*Vấn đề ở v1*: LLM không biết phải truyền định dạng tham số nào (id là chuỗi, số, hay email?), và không rõ hàm trả về cái gì. Điều này dẫn đến việc LLM thường xuyên truyền sai tham số (`Action: {"tool": "db_query", "args": {"name": "Nguyen Van A"}}`).

**Version 2 (Precise)**:
```json
{
  "name": "db_query",
  "description": "Truy vấn cơ sở dữ liệu để lấy thông tin chi tiết của một học sinh và lịch sử nộp bài của họ. Yêu cầu truyền đúng 'student_id' (kiểu string). Ví dụ args: {\"student_id\": \"2A202600928\"}",
  "parameters": {
    "type": "object",
    "properties": {
      "student_id": {
        "type": "string",
        "description": "Mã số sinh viên (student_id)"
      }
    },
    "required": ["student_id"]
  }
}
```
*Cải thiện ở v2*: Mô tả rất rõ chức năng, chỉ định rõ tham số bắt buộc là `student_id` và mô phỏng được schema JSON rõ ràng. Việc đưa ví dụ vào description giúp ReAct parsing chính xác hơn 99%.

### `model_eval` Tool

**Version 1 (Vague)**:
```json
{
  "name": "model_eval",
  "description": "Đánh giá bài làm."
}
```
*Vấn đề ở v1*: LLM không biết cách định dạng JSON cho rubric, thường xuyên chỉ truyền text trống hoặc thiếu tham số.

**Version 2 (Precise)**:
```json
{
  "name": "model_eval",
  "description": "Đánh giá submission text của học sinh dựa trên rubric. Trả về điểm số, feedback và phân tích chi tiết. Args yêu cầu: 'submission_text' (string) và 'rubric_json' (string - dạng JSON dump)."
}
```
*Cải thiện ở v2*: Yêu cầu định dạng rõ ràng, đặc biệt lưu ý `rubric_json` phải là dạng chuỗi JSON hợp lệ. LLM từ đó biết cách `json.dumps()` dict rubric trước khi truyền vào tool, tránh lỗi parsing.
