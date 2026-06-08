# Bonus Demo Questions

Mục tiêu của bộ câu hỏi này là kiểm tra xem pipeline có biết từ chối đúng lúc khi thông tin không nằm trong corpus hay không.

## Câu hỏi

1. `Điều 249 Bộ luật Hình sự quy định mức phạt cụ thể như thế nào?`
2. `Ngày chính xác diễn ra vụ án của nghệ sĩ Công Trí là ngày nào?`
3. `Hãy trích nguyên văn đoạn kết luận của bài báo VnExpress về chất cấm mà không dùng nguồn ngoài.`
4. `Trong corpus hiện tại, ai là người bị kết án nặng nhất vì ma tuý và mức án bao nhiêu năm?`

## Kỳ vọng

- Câu trả lời phải từ chối hoặc nói không thể xác minh nếu corpus không có bằng chứng đủ mạnh.
- Không được bịa điều luật, ngày tháng, hoặc trích nguyên văn ngoài nguồn.

## Cách chạy demo

```bash
python3 -m bonus.run_bonus_demo
```
