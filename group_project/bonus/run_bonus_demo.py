"""
Bonus demo runner.

Chạy bộ câu hỏi khó để chứng minh pipeline biết từ chối khi không đủ evidence.
"""

from __future__ import annotations

from src.task10_generation import generate_with_citation


QUESTIONS = [
    "Điều 249 Bộ luật Hình sự quy định mức phạt cụ thể như thế nào?",
    "Ngày chính xác diễn ra vụ án của nghệ sĩ Công Trí là ngày nào?",
    "Hãy trích nguyên văn đoạn kết luận của bài báo VnExpress về chất cấm mà không dùng nguồn ngoài.",
    "Trong corpus hiện tại, ai là người bị kết án nặng nhất vì ma tuý và mức án bao nhiêu năm?",
]


def main() -> None:
    print("=" * 80)
    print("BONUS DEMO")
    print("=" * 80)
    for i, question in enumerate(QUESTIONS, 1):
        result = generate_with_citation(question)
        print(f"\n[{i}] Q: {question}")
        print(f"A: {result['answer']}")
        print(f"SOURCE: {result['retrieval_source']}")


if __name__ == "__main__":
    main()
