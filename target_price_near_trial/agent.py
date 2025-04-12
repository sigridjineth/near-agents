import json
import requests
import logging

from nearai_langchain.orchestrator import NearAILangchainOrchestrator, RunMode

# --- 0. 오케스트레이터 초기화 ---
orchestrator = NearAILangchainOrchestrator(globals())
env = orchestrator.env

# run_mode가 LOCAL이면, 로컬 로그 설정
if orchestrator.run_mode == RunMode.LOCAL:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

logger = logging.getLogger(__name__)


# 로그 메시지 통일 함수
def log_message(message, level=logging.INFO):
    """
    run_mode에 따라 로컬 또는 env 로그로 기록하는 함수
    """
    if orchestrator.run_mode == RunMode.LOCAL:
        logger.log(level, message)
    else:
        env.add_agent_log(message, level)


# --- 1. 시스템 메시지 및 상수 정의 ---
SYSTEM_PROMPT = """
You are a NEAR price alert assistant.
You can set a target price in USD with a direction ('above' or 'below'), 
and notify the user when the current NEAR price meets that condition.
If the user hasn't set a target yet, request one.
"""

MODEL_NAME = "llama-v3p1-405b-instruct"
TARGET_FILE = "target_price.json"


# --- 2. 유틸 함수 ---


def load_target_info() -> dict:
    """
    target_price.json 파일에서 목표 가격 정보를 불러옵니다.
    형식: {"target": float, "direction": "above"|"below"}.
    파일이 없거나 JSON이 비정상이면 None을 반환합니다.
    """
    try:
        content = env.read_file(TARGET_FILE)  # NEAR AI env 방식
        if not content or content == "N/A":
            return None
        return json.loads(content)
    except FileNotFoundError:
        log_message(f"{TARGET_FILE} not found.", logging.WARNING)
        return None
    except json.JSONDecodeError as e:
        msg = f"Failed to parse {TARGET_FILE}: {e}"
        log_message(msg, logging.ERROR)
        env.add_message("assistant", f"[ERROR] {msg}")
        return None
    except Exception as e:
        msg = f"Unexpected error reading {TARGET_FILE}: {e}"
        log_message(msg, logging.ERROR)
        env.add_message("assistant", f"[ERROR] {msg}")
        return None


def save_target_info(data) -> None:
    """
    목표 가격 정보를 target_price.json에 저장합니다.
    data: dict or None
    """
    try:
        if data is None:
            # 목표를 초기화하려면, 파일을 빈 내용으로 덮어씌우거나 삭제할 수 있음
            env.write_file(TARGET_FILE, "")
            return
        env.write_file(TARGET_FILE, json.dumps(data))
    except Exception as e:
        msg = f"Failed to write {TARGET_FILE}: {e}"
        log_message(msg, logging.ERROR)
        env.add_message("assistant", f"[ERROR] {msg}")


def get_near_price() -> float:
    """
    CoinGecko API를 사용하여 현재 NEAR 가격(USD)을 가져옵니다.
    오류 발생 시 None 반환하고, 관련 메시지를 env.add_message로 알림.
    """
    try:
        url = "https://api.coingecko.com/api/v3/simple/price?ids=near&vs_currencies=usd"
        response = requests.get(url, timeout=10, verify=False)
        response.raise_for_status()
        data = response.json()

        if "near" in data and "usd" in data["near"]:
            return float(data["near"]["usd"])
        else:
            warn_msg = "[WARN] CoinGecko response missing 'near'/'usd' fields."
            log_message(warn_msg, logging.WARNING)
            env.add_message("assistant", warn_msg)
            return None
    except requests.exceptions.Timeout:
        err_msg = "[ERROR] Request to CoinGecko timed out."
        log_message(err_msg, logging.ERROR)
        env.add_message("assistant", err_msg)
        return None
    except requests.exceptions.RequestException as e:
        err_msg = f"[ERROR] RequestException from CoinGecko: {e}"
        log_message(err_msg, logging.ERROR)
        env.add_message("assistant", err_msg)
        return None
    except (json.JSONDecodeError, KeyError) as e:
        err_msg = f"[ERROR] Parsing CoinGecko response failed: {e}"
        log_message(err_msg, logging.ERROR)
        env.add_message("assistant", err_msg)
        return None


def parse_target_message(message: str) -> dict:
    """
    사용자 입력(예: '3.5 above')을 파싱하여
    {"target": float, "direction": "above"|"below"} 형태로 반환합니다.
    형식이 잘못되면 ValueError를 발생시킵니다.
    """
    parts = message.strip().split()
    if len(parts) != 2:
        raise ValueError("메시지는 '가격 방향' 형식이어야 합니다. 예) '3.5 above'")

    target_str, direction_str = parts
    direction_str = direction_str.lower()
    if direction_str not in ["above", "below"]:
        raise ValueError("두 번째 단어는 'above' 또는 'below' 여야 합니다.")

    if target_str.startswith("$"):
        target_str = target_str[1:]  # '$' 제거

    target_val = float(target_str)
    if target_val <= 0:
        raise ValueError("목표 가격은 0보다 커야 합니다.")

    return {"target": target_val, "direction": direction_str}


def generate_llm_response(user_messages, assistant_content: str) -> str:
    """
    시스템 프롬프트(SYSTEM_PROMPT) + 기존 메시지(user_messages) +
    새 assistant 메시지(assistant_content)를 합쳐 LLM에게 전달하고,
    최종 답변 텍스트를 반환합니다.
    """
    all_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *user_messages,
        {"role": "assistant", "content": assistant_content},
    ]
    # env.completion(model_name, messages) 사용
    return env.completion(MODEL_NAME, all_messages)


# --- 3. 메인 로직 ---


def handle_near_price_alert():
    """
    - 최근 사용자 메시지를 확인
    - 목표 가격 정보 (target_price.json) 불러오기
    - 만약 목표가 없으면, 사용자 입력을 파싱해 새로 설정
    - 이미 목표가 있다면 현재 NEAR 가격 체크 후, 조건 충족 시 알림 후 목표 초기화
    - 마지막으로 LLM 응답을 생성하고, 메시지를 보낸 뒤 다음 사용자 입력을 요청
    """
    messages = env.list_messages()
    if not messages or messages[-1]["role"] != "user":
        # 사용자 메시지가 없는 경우, 혹은 마지막 메시지가 사용자가 아니면 입력 요청
        log_message("No user message found, requesting user input...", logging.WARNING)
        env.request_user_input()
        return

    user_message_content = messages[-1]["content"]
    target_info = load_target_info()

    # LLM에 넘길 assistant 내용(추가 설명) 누적
    assistant_explanation = ""

    # 1) 목표 정보가 없을 경우 -> 사용자 입력을 통해 새로 설정 시도
    if not target_info:
        try:
            parsed = parse_target_message(user_message_content)
            save_target_info(parsed)
            assistant_explanation += (
                f"목표 가격이 설정되었습니다! NEAR가 ${parsed['target']:.2f} "
                f"{parsed['direction']}에 도달하면 알려드릴게요."
            )
        except ValueError as e:
            # 사용자 입력 형식 잘못됨
            err_msg = f"[WARN] {e}"
            log_message(err_msg, logging.WARNING)
            assistant_explanation += (
                f"{err_msg}\n올바른 예: '3.5 above' 또는 '$4 below' 형태"
            )
            final_answer = generate_llm_response(messages, assistant_explanation)
            env.add_message("assistant", final_answer)
            env.request_user_input()
            return final_answer
        except Exception as e:
            err_msg = f"[ERROR] Unexpected error setting target: {e}"
            log_message(err_msg, logging.ERROR)
            assistant_explanation += err_msg
            final_answer = generate_llm_response(messages, assistant_explanation)
            env.add_message("assistant", final_answer)
            env.request_user_input()
            return final_answer

    else:
        # 2) 이미 목표가 설정되어 있다면 -> 현재 가격 체크
        current_price = get_near_price()
        if current_price is None:
            # 오류 메시지는 get_near_price 내에서 이미 전송
            assistant_explanation += (
                "NEAR 가격을 불러오지 못해 알림을 진행할 수 없습니다."
            )
        else:
            t_val = target_info["target"]
            direction = target_info["direction"]
            if (direction == "above" and current_price > t_val) or (
                direction == "below" and current_price < t_val
            ):
                # 조건 충족 -> 알림 후 목표 초기화
                assistant_explanation += (
                    f"🚨 알림: 현재 NEAR=${current_price:.4f} "
                    f"(목표=${t_val:.2f}, {direction}) 조건 충족!\n"
                    "목표를 초기화합니다. 새 목표가 필요하면 다시 말씀해주세요."
                )
                save_target_info(None)
            else:
                # 아직 미도달
                assistant_explanation += (
                    f"현재 NEAR=${current_price:.4f}, 목표=${t_val:.2f} {direction}. "
                    f"아직 조건에 도달하지 않았습니다."
                )

    # 3) 최종 LLM 답변 생성 및 전송
    final_answer = generate_llm_response(messages, assistant_explanation)
    env.add_message("assistant", final_answer)

    # 4) 추가 사용자 입력 요청
    env.request_user_input()

    return final_answer


# --- 4. 실행부 (예: LOCAL 모드에서 실행 시) ---
# --- 4. 실행부 (LOCAL 모드에서 인터랙티브 실행 예시) ---
if __name__ == "__main__":
    log_message("Starting NEAR price alert handler...", logging.INFO)

    if orchestrator.run_mode == RunMode.LOCAL:
        # 예시로 무한 루프 형태로 계속 사용자 입력을 받고 처리
        while True:
            handle_near_price_alert()

            # 사용자의 즉각적인 반응을 보기 위해 잠시 대기하거나,
            # 어떤 조건에서 break로 빠져나갈 수 있도록 구성 가능
            # 예: 사용자가 'exit' 라고 입력하면 종료:
            last_msg = env.get_last_message(role="user")
            if last_msg and last_msg.strip().lower() == "exit":
                log_message("User requested exit. Stopping loop.", logging.INFO)
                break
    else:
        # 로컬 모드가 아닌 경우 한 번만 실행
        handle_near_price_alert()
