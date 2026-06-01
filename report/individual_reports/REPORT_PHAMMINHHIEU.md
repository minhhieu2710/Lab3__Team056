# Individual Report: Lab 3 - Chatbot vs ReAct Agent

- **Student Name**: Phạm Minh Hiếu
- **Role**: Dev B (LLM Providers & Chatbot Baseline)
- **Date**: 2026-06-01

---

## I. Technical Contribution (15 Points)

Trong bài Lab này, với vai trò Dev B, tôi phụ trách kiến trúc giao tiếp với các LLM Models (Provider Switching) và xây dựng hệ thống Chatbot Baseline để làm mốc so sánh.

- **Modules Implementated**: 
  - [`src/core/llm_provider.py`](../../src/core/llm_provider.py): Abstract base class chuẩn hóa interface.
  - [`src/core/openai_provider.py`](../../src/core/openai_provider.py) & [`src/core/gemini_provider.py`](../../src/core/gemini_provider.py): Implement gọi API và tracking token usage/latency.
  - [`chatbot.py`](../../chatbot.py): Kịch bản benchmark đánh giá năng lực của chatbot truyền thống với 7 test cases từ dễ đến khó (multi-step reasoning).
- **Code Highlights**: 
  Tôi đã chủ động bổ sung hàm `extract_json` trong `LLMProvider` để tiền xử lý chuỗi trả về từ LLM (bóc tách JSON khỏi markdown block), dọn đường cho Dev C parse Action dễ dàng hơn.
  ```python
  @staticmethod
  def extract_json(response_text: str) -> str:
      import re
      match = re.search(r"```(?:json)?(.*?)```", response_text, re.DOTALL)
      if match:
          return match.group(1).strip()
      return response_text.strip()
  ```
- **Documentation**: 
  Hệ thống Providers của tôi cung cấp một interface thống nhất `generate(prompt, system_prompt)` trả về dict chứa `content`, `usage`, `latency_ms`. Điều này cho phép ReAct Agent (do Dev C viết) có thể dễ dàng chuyển đổi giữa OpenAI, Gemini, hoặc Local Model qua biến môi trường `DEFAULT_PROVIDER` mà không cần sửa code logic.

---

## II. Debugging Case Study (10 Points)

- **Problem Description**: Khi chạy các test case đa bước (ví dụ: yêu cầu kiểm tra hàng tồn kho sau đó tính tổng bill) trên `chatbot.py`, model liên tục trả về các thông tin tồn kho và giá cả **bịa đặt** thay vì từ chối trả lời do thiếu kết nối hệ thống.
- **Log Source**: (Trích xuất từ `logs/chatbot_baseline_results.json`)
  ```json
  "prompt": "Check if the 'Macbook Pro M3' is currently in stock in our inventory system...",
  "response": "Yes, the Macbook Pro M3 is in stock. Applying the SAVE20 coupon...",
  "classification": {
    "status": "FAILURE",
    "failure_type": "hallucination",
    "analysis": "Chatbot BỊA DỮ LIỆU (hallucination)."
  }
  ```
- **Diagnosis**: 
  LLM được huấn luyện để luôn cố gắng cung cấp câu trả lời (helpfulness over truthfulness). Khi không có quyền truy cập vào Tools, chatbot truyền thống không nhận thức được giới hạn của nó trong không gian dữ liệu real-time. Do đó, model đã dựa vào kiến thức học được (hoặc tự sinh) để tạo ra một luồng phản hồi trông có vẻ logic nhưng sai sự thật.
- **Solution**: 
  Đây là một hạn chế bẩm sinh (architectural limitation) của kiến trúc Chatbot. Giải pháp triệt để không nằm ở việc sửa prompt, mà là phải nâng cấp lên kiến trúc **ReAct Agent**. Thông qua bước thiết lập Baseline này, nhóm đã có bằng chứng rõ ràng (empirical evidence) để chứng minh sự cần thiết của Agentic System (Giai đoạn 3). Ngoài ra, đối với lỗi parser khi ReAct Agent chạy (khi LLM trả về markdown block), tôi đã fix từ bên dưới tầng Provider bằng hàm regex `extract_json` như đã đề cập ở phần I.

---

## III. Personal Insights: Chatbot vs ReAct (10 Points)

- **Khối `Thought` giúp agent reasoning tốt hơn chatbot thế nào?**
  Khối `Thought` đóng vai trò như một bộ não trung gian (scratchpad). Thay vì sinh câu trả lời ngay lập tức, LLM buộc phải suy nghĩ từng bước (Chain-of-Thought). Điều này giúp nó break down vấn đề (ví dụ: "Bước 1: Tra tồn kho. Bước 2: Lấy giá."). Nó biến một tác vụ phức tạp thành một chuỗi các tác vụ vi mô có thể kiểm soát.
- **Trường hợp nào agent KÉM hơn chatbot?**
  Agent KÉM hơn chatbot trong các câu hỏi đơn giản không cần dữ liệu ngoài (Simple Q&A) hoặc trò chuyện xã giao (Chit-chat). Trong những trường hợp này, vòng lặp ReAct sinh ra overhead lớn (tốn nhiều token cho system prompt, parser, và loop) và latency cao gấp đôi, gấp ba so với việc sinh câu trả lời trực tiếp như Chatbot.
- **Observation (phản hồi từ môi trường) ảnh hưởng đến bước tiếp theo ra sao?**
  Observation chính là "đôi mắt" của Agent. Nó cung cấp Ground Truth (sự thật nền). Khi Agent đưa ra Action và nhận lại Observation (VD: lỗi Tool Not Found, hoặc Giá trị Tồn Kho = 0), nó sẽ đọc thông tin này trong lần lặp tiếp theo để điều chỉnh `Thought` (VD: "Món này hết hàng rồi, mình phải báo cho user thay vì tính tiền"). Đây là chìa khóa để Agent khắc phục hoàn toàn hiện tượng Hallucination mà Chatbot gặp phải.

---

## IV. Future Improvements (5 Points)

- **Scalability**: Xây dựng cơ chế Caching ở lớp Provider (Redis/Memcached). Nếu Agent hỏi cùng một câu với cùng Observation, ta có thể trả về cache thay vì gọi API LLM tốn tiền.
- **Safety**: 
  - Bổ sung Timeout cơ bản cho lớp API Provider để tránh việc Agent bị treo (hang) khi chờ Google/OpenAI API trả lời quá lâu.
  - Implement PII Scrubbing (xóa thông tin cá nhân) trước khi đẩy prompt lên các Cloud LLMs.
- **Performance**: 
  - Tích hợp Streaming Response (`stream()` method đã được khai báo nhưng chưa dùng trong Agent) để giảm perceived latency (thời gian chờ cảm nhận) cho người dùng cuối. 
  - Sử dụng local model nhẹ (như Phi-3) chuyên cho việc route/chọn tool để giảm chi phí token, chỉ dùng mô hình lớn như GPT-4o cho tác vụ suy luận cực kỳ phức tạp.
