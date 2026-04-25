"""
보호자용 K-CBCL 보고서 이해 지원 챗봇 (PoC)

기획안 3.2 기능 2의 최소 구현체.
- Google Gemini API 호출
- 시스템 프롬프트에 5원칙 + 보고서 컨텍스트 내장
- CLI 기반 대화 인터페이스
- 대화 히스토리 JSON 파일 저장 (세션 메모리 시늉)
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv

from system_prompt import build_system_prompt


# ===== 환경 설정 =====
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("❌ .env 파일에 GEMINI_API_KEY 설정되지 않았습니다.")
    sys.exit(1)

genai.configure(api_key=API_KEY)
MODEL_NAME  = "gemini-2.5-flash"  # 기획안 §4.3에 따라 Sonnet 단일 사용

SESSIONS_DIR = Path(__file__).parent / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)


# ===== 세션 관리 =====
def load_session(session_id: str) -> list[dict]:
    """이전 대화 히스토리를 로드한다. 첫 세션이면 빈 리스트 반환."""
    session_file = SESSIONS_DIR / f"{session_id}.json"
    if session_file.exists():
        with open(session_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_session(session_id: str, history: list[dict]) -> None:
    """대화 히스토리를 JSON 파일로 저장."""
    session_file = SESSIONS_DIR / f"{session_id}.json"
    with open(session_file, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
        
        
def history_to_gemini_format(history: list[dict]) -> list[dict]:
    """
    내부 히스토리(role: user/assistant)를 Gemini 포맷(role: user/model)으로 변환.
    """
    return [
        {
            "role": "user" if msg["role"] == "user" else "model",
            "parts": [msg["content"]],
        }
        for msg in history
    ]


# ===== Gemini API 호출 =====

# ===== 후처리 가드레일 =====

# 차단 대상 키워드 (정신과 진단명·치료 권고)
BLOCKED_KEYWORDS = [
    "ADHD", "주의력결핍", "주의력 결핍",
    "우울장애", "우울증",
    "불안장애", "불안 장애",
    "자폐", "자폐스펙트럼", "자폐 스펙트럼",
    "조현병", "양극성", "조울",
    "약물치료", "약물 치료",
    "처방", "처방받",
]


def contains_blocked_keyword(text: str) -> tuple[bool, str | None]:
    """
    응답에 차단 대상 키워드가 포함되었는지 확인한다.
    
    실제 프로덕션에서는:
    - 부정문 맥락 처리 (예: "ADHD가 아닙니다"는 OK)
    - 의미 기반 분류 (LLM 기반 검증)
    - 키워드 사전 동적 업데이트
    """
    for keyword in BLOCKED_KEYWORDS:
        if keyword in text:
            return True, keyword
    return False, None


def post_process_response(text: str) -> str:
    """
    LLM 응답 후처리.
    
    - 마스킹 토큰 정리: [CHILD_NAME] 같은 토큰이 응답에 노출되면 "자녀분"으로 치환
    - 마크다운 제거: **, * 같은 강조 표시 제거
    """
    # 마스킹 토큰 노출 방지 (이중 안전장치)
    text = text.replace("[CHILD_NAME]", "자녀분")
    
    # 마크다운 제거 (시스템 프롬프트에 금지했지만 이중 안전장치)
    text = text.replace("**", "").replace("__", "")
    
    return text


def chat(history: list[dict], user_message: str, system_prompt: str) -> str:
    """사용자 메시지에 대한 챗봇 응답 생성."""
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=system_prompt,
    )

    chat_session = model.start_chat(history=history_to_gemini_format(history))
    response = chat_session.send_message(user_message)
    assistant_message = response.text

        # 후처리 1: 키워드 차단 검사
    blocked, keyword = contains_blocked_keyword(assistant_message)
    if blocked:
        # PoC 단순 구현: 차단 시 안전 응답으로 교체
        # 실제 운영 시에는 재생성 시도 또는 상담사 안내로 분기
        assistant_message = (
            "이 부분은 임상적 판단이 필요한 영역이라, "
            "상담사님과 직접 이야기 나누시는 것이 좋겠어요. "
            "지금 떠오르시는 다른 궁금한 점이 있으시면 함께 정리해볼까요?"
        )
        print(f"\n[가드레일 작동: '{keyword}' 키워드 감지 → 안전 응답으로 교체]")

    # 후처리 2: 포매팅 정리
    assistant_message = post_process_response(assistant_message)

    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": assistant_message})

    return assistant_message


# ===== CLI 메인 루프 =====
def main():
    print("=" * 60)
    print("  K-CBCL 보고서 이해 지원 챗봇 (PoC)")
    print("=" * 60)
    print()
    print("아동 심리 검사 결과에 대해 궁금하신 점을 편하게 물어보세요.")
    print("종료하려면 'quit' 또는 'exit'를 입력하세요.")
    print()

    session_id = input("세션 ID (예: parent001, 처음이면 새 ID 입력): ").strip()
    if not session_id:
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"세션 ID 자동 생성: {session_id}")

    history = load_session(session_id)
    if history:
        print(f"\n[이전 대화 {len(history) // 2}턴이 복원되었습니다.]")
    else:
        print("\n[새 대화를 시작합니다.]")

    print()
    system_prompt = build_system_prompt()

    while True:
        try:
            user_input = input("\n👤 보호자: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n[대화를 종료합니다. 세션이 저장되었습니다.]")
            save_session(session_id, history)
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "종료"):
            print("\n[대화를 종료합니다. 세션이 저장되었습니다.]")
            save_session(session_id, history)
            break

        try:
            response = chat(history, user_input, system_prompt)
            print(f"\n🤖 챗봇:\n{response}")
            save_session(session_id, history)
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")


if __name__ == "__main__":
    main()