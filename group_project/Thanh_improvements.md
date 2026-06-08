# Cải Tiến G4 + G5 — Nguyễn Đức Thành (2A202600838)

## Các File Đã Thay Đổi

| File | Loại thay đổi |
|------|--------------|
| `agents/rag_manager.py` | Refactor + bug fix |
| `agents/orchestrator.py` | Clarify comment |
| `scripts/ingest_sources.py` | Viết mới hoàn toàn |

---

## 1. `scripts/ingest_sources.py` — Viết mới

**Vấn đề cũ:** File chỉ là stub, chỉ in danh sách sources, không làm gì thực tế.

**Thay đổi:**
- Copy các file `.md` từ `personal_project/2A202600838_NguyenDucThanh/data/standardized/` vào `group_project/data/source_docs/`
- Tự động skip file đã tồn tại cùng stem (idempotent — chạy nhiều lần không bị duplicate)
- Tự verify sau khi ingest bằng cách load corpus và in danh sách documents

**Kết quả:** 2 file unique của Thành được thêm vào corpus nhóm:
- `legal/73luat.md`
- `legal/huong-dan-luat-phong-chong-ma-tuy.md`

Corpus nhóm tăng từ ~13 lên **19 documents**.

---

## 2. `agents/rag_manager.py` — 6 cải tiến

### Fix #1 — Xóa double citation

**Vấn đề:** `_append_citation_footer()` ghi "Nguồn: [1]..." trực tiếp vào phần `answer`, trong khi `main_app.py` đã render một block citation riêng từ `response.citations`. Người dùng thấy citation 2 lần.

**Fix:** Bỏ lời gọi `_append_citation_footer` trong `answer_chat`. UI (`main_app.py`) tự handle citation.

---

### Fix #2 — Xóa mutable side-effect state

**Vấn đề:** `_call_openai()` set `self._last_generation_mode` như một side-effect, rồi `answer_chat()` đọc lại sau đó. Không thread-safe — hai request concurrent sẽ đọc nhầm giá trị nhau.

**Fix:** `_call_openai()` trả về tuple `(answer: str | None, mode: str)`. `_generate_answer()` cũng trả về `(str, str)`. Không còn state mutable trên instance.

```python
# Trước
def _call_openai(...) -> str | None:
    self._last_generation_mode = "openai"  # side-effect

# Sau
def _call_openai(...) -> tuple[str | None, str]:
    return answer, "openai"
```

---

### Fix #3 — Lazy init PostgresVectorStore

**Vấn đề:** `__init__` luôn tạo `PostgresVectorStore()` dù `data_mode = "personal"`, dẫn đến đọc env vars DB không cần thiết mỗi lần khởi tạo.

**Fix:** Dùng `@property` lazy — chỉ khởi tạo `PostgresVectorStore` khi thực sự truy cập lần đầu.

```python
@property
def _postgres_store(self):
    if self._postgres_store_instance is None:
        from .postgres_store import PostgresVectorStore
        self._postgres_store_instance = PostgresVectorStore()
    return self._postgres_store_instance
```

---

### Fix #4 — Tăng snippet width

**Vấn đề:** `shorten(..., width=240)` cắt snippet quá ngắn, LLM nhận context không đủ để trả lời chính xác.

**Fix:** Tăng lên `800` ký tự (có thể override qua env `GROUP_RAG_SNIPPET_WIDTH`).

**Xác nhận:** `snippet_0 len: 800`

---

### Fix #5 — Cắt history content trong follow-up query

**Vấn đề:** `_build_query` nhét toàn bộ `prior_assistant[-1]` (có thể 1000+ ký tự) vào retrieval query, làm loãng tín hiệu tìm kiếm.

**Fix:** Cắt tối đa `HISTORY_CONTEXT_CHARS = 300` ký tự (có thể override qua env `GROUP_RAG_HISTORY_CHARS`).

```python
truncated = shorten(prior_assistant[-1], width=HISTORY_CONTEXT_CHARS, placeholder="...")
```

---

### Fix #6 — Score threshold

**Vấn đề:** Document có score 0.01 được đưa vào context LLM như document có score 0.9 — gây nhiễu câu trả lời.

**Fix:** Filter bỏ kết quả có `score < MIN_SCORE` (mặc định `0.05`, override qua env `GROUP_RAG_MIN_SCORE`).

```python
ranked = [r for r in ranked if (r.get("score") or 0.0) >= min_score]
```

**Xác nhận:** `score_0: 0.96`

---

## Env Variables Mới (tùy chỉnh không cần sửa code)

| Biến | Mặc định | Mô tả |
|------|----------|-------|
| `GROUP_RAG_MIN_SCORE` | `0.05` | Ngưỡng score tối thiểu để nhận document |
| `GROUP_RAG_SNIPPET_WIDTH` | `800` | Độ dài snippet gửi lên LLM |
| `GROUP_RAG_HISTORY_CHARS` | `300` | Số ký tự tối đa của history context trong follow-up query |
| `GROUP_RAG_TOP_K` | `5` | Số document retrieve |
| `GROUP_RAG_TEMPERATURE` | `0.25` | Temperature OpenAI |
| `GROUP_RAG_DATA_MODE` | `personal` | `personal` dùng local corpus, `db` dùng pgvector |
