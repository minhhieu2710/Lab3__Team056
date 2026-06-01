# Báo cáo Giai đoạn 1: Chatbot Baseline

## 1. Kết quả chạy Chatbot với các Test Case

Chúng tôi đã thiết kế 3 test case để kiểm tra khả năng suy luận và sử dụng công cụ của Chatbot Baseline (không có khả năng gọi hàm/tool):

### Test Case 1: Simple Math Q&A
- **Prompt**: "What is 25 * 4 + 10?"
- **Kết quả**: Chatbot trả lời chính xác là 110.
- **Đánh giá**: Thành công. LLM có khả năng thực hiện các phép toán cơ bản tốt nhờ vào lượng kiến thức toán học đã được pre-train.

### Test Case 2: E-commerce multi-step reasoning
- **Prompt**: "I want to buy a laptop that costs $1200. I have a 15% discount coupon. The shipping fee is $20 based on my location. Can you calculate the final total for me?"
- **Kết quả**: Chatbot trả lời chính xác: Tính giảm giá ($1200 * 15% = $180), giá sau giảm ($1020), cộng phí ship ($1040).
- **Đánh giá**: Thành công. Chatbot có thể làm các bài toán suy luận nhiều bước đơn giản nếu dữ liệu đã được cung cấp sẵn trong prompt.

### Test Case 3: Tool-dependent inquiry
- **Prompt**: "Check if the 'Macbook Pro M3' is in stock in the 'check_stock' system, and if so, calculate the final price with a 10% discount and $15 shipping fee."
- **Kết quả**: Thất bại (Hallucination). Chatbot tự bịa ra một câu trả lời như "The Macbook Pro M3 is in stock in our system with 12 units left..." hoặc "I don't have access to the check_stock system...".
- **Đánh giá**: Chatbot hoàn toàn không có khả năng tra cứu dữ liệu thực tế từ database hay hệ thống ngoài.

## 2. Giới hạn của Chatbot Baseline

Dựa trên kết quả thử nghiệm, chúng tôi rút ra các giới hạn của một LLM Chatbot thuần túy (không có Agentic behavior):
1. **Thiếu khả năng truy xuất dữ liệu động**: Không thể tương tác với cơ sở dữ liệu, API, hay hệ thống nội bộ (như `check_stock`). Dẫn đến việc mô hình sẽ từ chối trả lời hoặc nghiêm trọng hơn là **ảo giác (hallucinate)** thông tin.
2. **Không có khả năng tương tác với môi trường**: Không thể thực thi code, gửi email, hay ghi vào cơ sở dữ liệu.
3. **Giới hạn trong suy luận phức tạp**: Đối với các bài toán cần tra cứu trung gian (ví dụ: lấy giá trị A từ DB, sau đó dùng A để tính B), LLM không thể làm được vì thiếu quan sát (Observation) từ môi trường.
