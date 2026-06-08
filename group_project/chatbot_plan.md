# Kế Hoạch Nhóm - RAG Chatbot

## Lý Do Chọn Chatbot

Nhóm chọn hướng **RAG Chatbot** để demo một sản phẩm có thể dùng được ngay, tập trung vào:

- Chat UI rõ ràng.
- Câu trả lời có citation.
- Hỗ trợ follow-up bằng memory.
- Hiển thị source documents để người chấm kiểm tra nguồn.

Nhánh evaluation pipeline giữ lại trong repo để tham khảo, nhưng không phải deliverable chính.

## Thành Viên

| Thành viên | MSSV |
|------------|------|
| Đặng Minh Hải | 2A202600713 |
| Nguyễn Đức Thành | 2A202600838 |
| Hoàng Phúc Quân | 2A202600560 |

## Phân Công

| ID | Owner | Task | Deliverable | Done khi |
|----|-------|------|-------------|----------|
| G1 | Hải | Setup base project | Khung `group_project/app.py` và package chatbot | Chạy được app rỗng, không lỗi import |
| G2 | Hải | Setup UI | Chat layout, history, reset, render answer/source | Chat nhiều lượt trong một session |
| G3 | Hải | Citation rendering và session bridge | Chuẩn hóa message format, render citation, nối session state vào backend | Citation hiển thị rõ và follow-up truyền đúng history |
| G4 | Thành | RAG adapter | Hàm `answer_chat(question, history)` | Trả về đúng schema answer + sources + metadata |
| G5 | Thành | Backend follow-up support | Logic dùng history để hiểu câu hỏi tiếp theo | Follow-up retrieve đúng evidence |
| G6 | Quân | Source documents panel | Format/dedup source docs cho UI | Có title/source/link/snippet rõ ràng |
| G7 | Quân | Demo questions và QA | Bộ câu hỏi demo + checklist test | Có câu hỏi thường, follow-up, thiếu evidence, citation |
| G8 | Cả nhóm | Final integration | App chạy end-to-end | Demo được ít nhất 5 câu hỏi |

## Luồng Làm

### Phase 1

- Hải dựng skeleton app và layout.
- Thành chuẩn hóa interface trả kết quả từ pipeline cá nhân.
- Quân chuẩn bị demo questions và source metadata cần hiển thị.

### Phase 2

- Thành nối retrieval/generation thật vào adapter.
- Hải thay mock response bằng call backend thật.
- Quân hoàn thiện source panel và dedup data.

### Phase 3

- Hải hoàn thiện history và citation display.
- Thành tối ưu follow-up handling ở backend.
- Quân test các câu khó và các case thiếu evidence.

### Phase 4

- Cả nhóm chạy demo cùng môi trường.
- Fix lỗi import, path, env.
- Chốt kịch bản demo ổn định.

## Contract Gợi Ý

```python
def answer_chat(question: str, history: list[dict]) -> dict:
    return {
        "answer": "...",
        "citations": [
            {
                "id": "S1",
                "title": "...",
                "source": "...",
                "url": "...",
            }
        ],
        "sources": [
            {
                "id": "S1",
                "title": "...",
                "source": "...",
                "url": "...",
                "snippet": "...",
                "score": 0.0,
            }
        ],
        "metadata": {
            "used_memory": True,
            "retrieval_mode": "hybrid",
        },
    }
```

## Command Chạy Dự Kiến

```bash
pip install -r requirements.txt
streamlit run group_project/app.py
```
