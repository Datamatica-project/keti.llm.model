# test_inference.py
from utils.inference import generate_response  # inference.py가 같은 폴더에 있다는 가정

if __name__ == "__main__":
    question = "마늘 재배 시기 알려줘"
    result = generate_response(question, session_id="test-session")

    print("=== 답변 ===")
    print(result["answer"])
    print()
    print("=== 참고 문서 ===")
    for r in result.get("rank", []):
        print(f"- {r[0].get('document')} | score={r[1]}")
