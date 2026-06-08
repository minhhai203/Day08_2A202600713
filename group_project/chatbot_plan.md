# Kế Hoạch Nhóm - RAG Chatbot

## Lý Do Chọn Chatbot

Nhóm chọn hướng **RAG Chatbot** để demo một sản phẩm có thể dùng được ngay, tập trung vào:

- Chat UI rõ ràng.
- Câu trả lời có citation.
- Hỗ trợ follow-up bằng memory.
- Hiển thị source documents để người chấm kiểm tra nguồn.

Nhánh evaluation pipeline giữ lại trong repo để tham khảo, nhưng không phải deliverable chính.

## Tinh Hoa Nên Mang Sang Từ Repo Mẫu

### Nên giữ

- **Chainlit** làm chat shell chính vì hỗ trợ streaming, actions, avatar, feedback, và session state tốt.
- **Agno Agent** làm lớp orchestration để gom prompt, memory, tools, và source attribution.
- **PostgresDb** làm session store nếu muốn giữ lịch sử hội thoại theo user/session.
- **PgVector + Knowledge** làm backend vector RAG cho tài liệu đã crawl.
- **Discovery tools** để list/read source docs khi cần kiểm tra hoặc hiển thị nguồn.
- **Custom CSS / avatars** nếu dùng Chainlit để làm UI trông gọn và có nhận diện.

### Không cần bê nguyên

- SQL agent và biểu đồ.
- Dashboard trace riêng cho nội bộ nếu team không đủ thời gian.
- Docling pipeline OCR nặng nếu nguồn chính của bài đã là markdown/text sạch từ crawl.
- Multi-agent team phức tạp kiểu SQL + Doc + Orchestrator, vì bài nhóm chatbot chỉ cần một trục RAG rõ ràng.

## Kiến Trúc Đề Xuất

```text
group_project/
├── agents/
│   ├── main_app.py        # Chainlit entrypoint
│   ├── orchestrator.py    # Agent chính điều phối hội thoại
│   ├── rag_manager.py     # PgVector, Knowledge, ingest/search
│   ├── discovery_tools.py # List/read source docs
│   ├── source_view.py     # Format citation/source panel
│   └── memory_store.py    # Session/history helpers
├── public/                # Runtime assets Chainlit serve từ root repo
│   ├── custom.css         # Chỉnh giao diện Chainlit
│   └── avatars/           # Avatar chat nếu cần
├── data/
│   └── source_docs/       # Tài liệu nguồn để crawl / index / demo
├── scripts/
│   ├── ingest_sources.py  # Nạp dữ liệu vào vector DB
│   └── reindex_sources.py  # Rebuild index khi cần
└── docs/
    └── demo_questions.md  # Bộ câu hỏi demo + follow-up
```

## UI Và Công Nghệ Nên Dùng

- **Primary UI:** `Chainlit`.
- **Storage:** `PostgresDb` cho session, `pgvector` cho tri thức.
- **Embeddings:** `OpenAI` là ưu tiên nếu env đã có; fallback thì mới dùng model khác.
- **Retrieval:** `Agno Knowledge` trên `PgVector`.
- **Source display:** panel/section trong Chainlit message hoặc sidebar.
- **Tracing:** chỉ để debug nội bộ, không phải core deliverable.

## Thành Viên

| Thành viên | MSSV |
|------------|------|
| Đặng Minh Hải | 2A202600713 |
| Nguyễn Đức Thành | 2A202600838 |
| Hoàng Phúc Quân | 2A202600560 |

## Phân Công

| ID | Owner | Task | Deliverable | Done khi |
|----|-------|------|-------------|----------|
| G1 | Hải | Layout shell | Tạo `agents/main_app.py`, nạp `public/custom.css`, khởi tạo session UI | App mở được, layout khung sẵn sàng cho team cắm logic |
| G2 | Hải | Chat rendering | Hiển thị message user/assistant, reset chat, history panel | Chat nhiều lượt trong một session |
| G3 | Hải | Citation bridge | Render citation/source block từ response schema | Citation hiển thị rõ, không vỡ format |
| G4 | Thành | Orchestrator / RAG adapter | Tạo `agents/orchestrator.py` và `agents/rag_manager.py` | Có hàm trả `answer_chat(question, history)` rõ schema |
| G5 | Thành | Retrieval pipeline | Nối embedding, vector search, follow-up logic | Follow-up retrieve đúng evidence |
| G6 | Quân | Source tools | Tạo `agents/discovery_tools.py` + `docs/demo_questions.md` | Có list/read nguồn và bộ câu hỏi demo chuẩn |
| G7 | Quân | Source panel QA | Format/dedup source docs, kiểm tra source snippets | Có title/source/link/snippet rõ ràng |
| G8 | Cả nhóm | Final integration | App chạy end-to-end | Demo được ít nhất 5 câu hỏi |

## Luồng Làm

### Phase 1

- Hải dựng skeleton Chainlit app và layout.
- Thành chuẩn hóa interface trả kết quả từ pipeline cá nhân.
- Quân chuẩn bị demo questions và source metadata cần hiển thị.
- Cả nhóm thống nhất một schema response chung trước khi code sâu.

### Phase 2

- Thành nối retrieval/generation thật vào adapter hoặc orchestrator.
- Hải thay mock response bằng call backend thật.
- Quân hoàn thiện source panel và dedup data.
- Mỗi người chỉ làm trong file ownership của mình, hạn chế đụng chéo.

### Phase 3

- Hải hoàn thiện history và citation display.
- Thành tối ưu follow-up handling ở backend.
- Quân test các câu khó và các case thiếu evidence.
- Nếu thay schema, phải cập nhật cùng lúc `contracts.py` và `demo_questions.md`.

## File Ownership

| File / Folder | Owner chính |
|---------------|-------------|
| `group_project/agents/main_app.py` | Hải |
| `group_project/agents/memory_store.py` | Hải |
| `group_project/agents/source_view.py` | Hải |
| `public/` | Hải |
| `group_project/agents/orchestrator.py` | Thành |
| `group_project/agents/rag_manager.py` | Thành |
| `group_project/scripts/ingest_sources.py` | Thành |
| `group_project/agents/discovery_tools.py` | Quân |
| `group_project/docs/demo_questions.md` | Quân |
| `group_project/scripts/reindex_sources.py` | Quân |
| `group_project/agents/contracts.py` | Hải + Thành cùng giữ schema |

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
chainlit run group_project/agents/main_app.py
```
