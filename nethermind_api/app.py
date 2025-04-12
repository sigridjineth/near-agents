import os
import httpx
from fastapi import FastAPI, HTTPException, Path, Query
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# --- Configuration ---
NETHERMIND_RPC_URL = "https://eth-sepolia.public.blastapi.io"

# --- Pydantic Models ---
class CallTraceItem(BaseModel):
    from_addr: str = Field(..., alias="from")
    to_addr: Optional[str] = Field(None, alias="to")
    input_data: str = Field(..., alias="input")
    output_data: Optional[str] = Field(None, alias="output")
    gas_used: str = Field(..., alias="gasUsed")
    call_type: str = Field(..., alias="type")
    value: Optional[str] = None # Ether 전송량 (있는 경우)
    error: Optional[str] = None
    calls: Optional[List['CallTraceItem']] = None

CallTraceItem.model_rebuild()

class AnalysisResponse(BaseModel):
    transaction_hash: str
    status: str
    error_message: Optional[str] = None
    call_trace_summary: str
    mermaid_explanation: str # Mermaid 다이어그램 설명 추가
    mermaid_diagram: Optional[str] = None # Mermaid 다이어그램 문자열 추가
    raw_trace: Optional[Dict[str, Any]] = None

# --- FastAPI App ---
app = FastAPI(
    title="Nethermind Transaction Analyzer with Mermaid",
    description="Analyzes Ethereum transactions using Nethermind's trace_transaction (callTracer), provides a summary, and generates a Mermaid sequence diagram.",
    version="0.2.0",
)

# --- Helper Functions ---

async def fetch_trace(tx_hash: str) -> Dict[str, Any]:
    """Nethermind 노드에 trace_transaction 요청을 보냅니다."""
    payload = {
        "jsonrpc": "2.0",
        "method": "trace_transaction",
        "params": [
            tx_hash,
            {"tracer": "callTracer", "tracerConfig": {"withLog": False}} # 로그는 제외하여 트레이스 간소화
        ],
        "id": 1
    }
    try:
        async with httpx.AsyncClient(timeout=60.0) as client: # 복잡한 트레이스를 위해 타임아웃 증가
            response = await client.post(NETHERMIND_RPC_URL, json=payload)
            response.raise_for_status()
            result_json = response.json()

            if "error" in result_json:
                raise HTTPException(
                    status_code=400,
                    detail=f"Nethermind RPC Error: {result_json['error']['message']}"
                )
            if "result" not in result_json:
                 raise HTTPException(
                    status_code=500,
                    detail="Invalid response from Nethermind: 'result' field missing."
                )
            # 결과가 null인 경우도 처리 (트랜잭션이 없거나 trace 불가)
            if result_json["result"] is None:
                 raise HTTPException(
                     status_code=404,
                     detail=f"Trace result is null for transaction {tx_hash}. Ensure the transaction exists and the node has trace capabilities enabled."
                 )
            return result_json["result"]

    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Could not connect to Nethermind node at {NETHERMIND_RPC_URL}. Error: {exc}"
        )
    except Exception as e:
         if isinstance(e, HTTPException):
             raise e
         raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

def format_call_trace(trace_data: Dict[str, Any], indent: str = "") -> str:
    """callTracer 결과를 사람이 읽기 쉬운 문자열로 재귀적으로 포맷합니다."""
    # (이전 코드와 동일하게 유지 또는 필요시 약간 수정)
    summary = ""
    try:
        from_addr = trace_data.get("from", "N/A")
        to_addr = trace_data.get("to", "[Contract Creation]")
        input_data = trace_data.get("input", "0x")
        function_sig = input_data[:10] if len(input_data) >= 10 else input_data
        gas_used_hex = trace_data.get("gasUsed", "0x0")
        gas_used_int = int(gas_used_hex, 16) if gas_used_hex else 0
        call_type = trace_data.get("type", "UNKNOWN")
        error = trace_data.get("error")
        value_hex = trace_data.get("value", "0x0")
        value_int = int(value_hex, 16) if value_hex else 0

        summary += f"{indent}- {call_type} from {from_addr} to {to_addr}"
        if value_int > 0:
             summary += f" | Value: {value_int} wei ({value_hex})" # Ether 전송량 표시
        summary += "\n"

        summary += f"{indent}  Input (func sig): {function_sig}\n"
        summary += f"{indent}  Gas Used: {gas_used_int} ({gas_used_hex})\n"
        if error:
            summary += f"{indent}  Error: {error}\n"
        else:
            output = trace_data.get("output", "0x")
            output_display = output[:66] + "..." if len(output) > 66 else output
            summary += f"{indent}  Output: {output_display}\n"

        if "calls" in trace_data and trace_data["calls"]:
            for sub_call in trace_data["calls"]:
                summary += format_call_trace(sub_call, indent + "  ")

    except Exception as e:
        summary += f"{indent}  [Error formatting this call: {e}]\n"
        print(f"Error formatting call data: {trace_data}, Error: {e}")

    return summary

# --- Mermaid Diagram Generation ---

def get_short_address(address: Optional[str]) -> str:
    """주소를 짧게 만듭니다 (예: 0x1234...abcd)"""
    if not address:
        return "[Unknown]"
    if len(address) > 10:
        return f"{address[:6]}...{address[-4:]}"
    return address

def generate_mermaid_sequence(call_data: Dict[str, Any], alias_map: Dict[str, str], lines: List[str]):
    """Mermaid 시퀀스 다이어그램 라인을 재귀적으로 생성합니다."""
    try:
        from_addr = call_data.get("from")
        to_addr = call_data.get("to")
        call_type = call_data.get("type", "CALL")
        input_data = call_data.get("input", "0x")
        output_data = call_data.get("output", "0x")
        error = call_data.get("error")
        gas_used_hex = call_data.get("gasUsed", "0x0")
        gas_used_int = int(gas_used_hex, 16) if gas_used_hex else 0
        value_hex = call_data.get("value", "0x0")
        value_int = int(value_hex, 16) if value_hex else 0

        # 주소 별칭 가져오기
        from_alias = alias_map.get(from_addr, get_short_address(from_addr))
        to_alias = "NewContract" # 기본값 (생성 시)
        if to_addr:
            to_alias = alias_map.get(to_addr, get_short_address(to_addr))
        elif call_type != 'CREATE': # to가 없는데 CREATE가 아니면 이상함
             to_alias = "[Unknown Target]"

        # 함수 시그니처 또는 입력 데이터 간략 표시
        label = call_type
        if input_data and input_data != "0x":
            label += f"({input_data[:10]}..)" # 함수 시그니처
        if value_int > 0:
             label += f" [{value_int} wei]" # Ether 전송량

        # 호출 화살표
        lines.append(f"    {from_alias} ->> {to_alias}: {label}")
        lines.append(f"    activate {to_alias}")

        # 가스 사용량 노트
        lines.append(f"    Note right of {to_alias}: Gas: {gas_used_int}")

        # 하위 호출 재귀 처리
        if "calls" in call_data and call_data["calls"]:
            for sub_call in call_data["calls"]:
                generate_mermaid_sequence(sub_call, alias_map, lines)

        # 반환 화살표 (성공/실패)
        if error:
            lines.append(f"    Note right of {to_alias}: Error: {error}")
            lines.append(f"    {to_alias} -->> {from_alias}: Failed: {error}")
        else:
            output_display = output_data[:10] + "..." if len(output_data) > 10 else output_data
            lines.append(f"    {to_alias} -->> {from_alias}: Success ({output_display})")

        lines.append(f"    deactivate {to_alias}")

    except Exception as e:
        lines.append(f"    Note over {from_alias},{to_alias}: Error generating part of diagram: {e}")
        print(f"Error generating mermaid sequence for call: {call_data}, Error: {e}")


def create_mermaid_diagram(trace_data: Dict[str, Any]) -> Optional[str]:
    """전체 트레이스 데이터로부터 Mermaid 시퀀스 다이어그램 문자열을 생성합니다."""
    if not trace_data:
        return None

    participants = set()
    alias_map = {}
    participant_counter = 0

    def find_participants(call):
        nonlocal participant_counter
        from_addr = call.get("from")
        to_addr = call.get("to")

        if from_addr and from_addr not in participants:
            participants.add(from_addr)
            alias = f"P{participant_counter}"
            # 첫번째 from은 EOA로 간주 (단순화)
            if participant_counter == 0:
                alias = "EOA"
            # elif call_type == "CREATE":
            #      # 생성된 컨트랙트 주소는 output에 있을 수 있음 (더 정확하게 하려면 추가 로직 필요)
            #      alias = f"Contract{chr(ord('A') + participant_counter -1)}"
            else:
                 alias = f"Contract{chr(ord('A') + participant_counter -1)}"
            alias_map[from_addr] = alias
            participant_counter += 1

        if to_addr and to_addr not in participants:
             participants.add(to_addr)
             alias = f"Contract{chr(ord('A') + participant_counter -1)}"
             alias_map[to_addr] = alias
             participant_counter += 1


        if "calls" in call and call["calls"]:
            for sub_call in call["calls"]:
                find_participants(sub_call)

    find_participants(trace_data) # 루트 호출부터 참여자 탐색

    mermaid_lines = ["sequenceDiagram"]
    # 참여자 선언 (별칭 사용)
    for addr, alias in alias_map.items():
         # 주소를 포함하여 명확성 증가
         mermaid_lines.append(f"    participant {alias} as {alias}<br/>({get_short_address(addr)})")


    # 시퀀스 생성 시작
    generate_mermaid_sequence(trace_data, alias_map, mermaid_lines)

    return "\n".join(mermaid_lines)

# --- API Endpoint ---

@app.get(
    "/analyze/transaction/{tx_hash}",
    response_model=AnalysisResponse,
    summary="Analyze Transaction Trace with Mermaid Diagram",
    description="Fetches the call trace, provides a summary, and generates a Mermaid sequence diagram.",
    tags=["Analysis"]
)
async def analyze_transaction(
    tx_hash: str = Path(
        ...,
        description="The hash of the transaction to analyze (e.g., 0x...)",
        min_length=66,
        max_length=66,
        pattern="^0x[a-fA-F0-9]{64}$"
    ),
    include_raw_trace: bool = Query(False, description="Include the raw trace result in the response")
):
    """
    지정된 트랜잭션 해시에 대한 호출 트레이스를 분석하고 Mermaid 시퀀스 다이어그램을 생성합니다.
    """
    print(f"Analyzing transaction: {tx_hash}")

    raw_trace_result = await fetch_trace(tx_hash)

    if not raw_trace_result:
         raise HTTPException(
             status_code=404,
             detail=f"Trace result not found or empty for transaction {tx_hash}."
         )

    # 상태 및 에러 메시지 분석
    status = "Success"
    error_message = None
    if "error" in raw_trace_result and raw_trace_result["error"]:
        status = "Failed"
        error_message = raw_trace_result["error"]
        print(f"Transaction {tx_hash} failed: {error_message}")

    # 텍스트 요약 생성
    summary = f"Trace for {tx_hash} ({status}):\n"
    summary += format_call_trace(raw_trace_result)

    # Mermaid 다이어그램 생성
    mermaid_diagram_str = None
    mermaid_explanation = "Mermaid sequence diagram could not be generated for this trace."
    try:
        mermaid_diagram_str = create_mermaid_diagram(raw_trace_result)
        if mermaid_diagram_str:
             mermaid_explanation = (
                 "Below is a Mermaid sequence diagram visualizing the internal calls of the transaction. "
                 "Arrows indicate calls between participants (EOA or Contracts). "
                 "Notes show gas usage and errors. Use a Mermaid renderer (e.g., online editors, integrated markdown viewers) to view the diagram."
             )
    except Exception as e:
        print(f"Error generating Mermaid diagram for {tx_hash}: {e}")
        # 다이어그램 생성 실패 시에도 에러를 반환하지 않고 설명만 업데이트


    return AnalysisResponse(
        transaction_hash=tx_hash,
        status=status,
        error_message=error_message,
        call_trace_summary=summary.strip(),
        mermaid_explanation=mermaid_explanation,
        mermaid_diagram=mermaid_diagram_str,
        raw_trace=raw_trace_result if include_raw_trace else None
    )

# --- Main execution ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
