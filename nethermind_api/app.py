import os
import httpx # Use synchronous httpx client
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, Path, Query
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
# import asyncio # No longer needed
import time
import json

# --- Configuration ---
NETHERMIND_RPC_URL = "https://sepolia.drpc.org"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "") # <-- 환경 변수에서만 읽도록 변경

if not GOOGLE_API_KEY:
    print("Warning: GOOGLE_API_KEY environment variable not set. LLM interpretation will be disabled.")
    genai_enabled = False
else:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        genai_enabled = True
    except Exception as e:
        print(f"Error configuring Google AI: {e}. LLM interpretation will be disabled.")
        genai_enabled = False


# --- Models ---
class AnalysisResponse(BaseModel):
    transaction_hash: str
    status: str
    error_message: Optional[str] = None
    call_trace_summary: str
    mermaid_explanation: str
    mermaid_diagram: Optional[str] = None
    raw_trace: Optional[Dict[str, Any]] = None
    interpretation: Optional[str] = None


# --- FastAPI App ---
app = FastAPI()

# === Updated function to call Google Gemini (Synchronous) ===
def get_llm_interpretation(summary: str) -> str: # <-- Remove async
    """
    Calls Google's Gemini model synchronously to interpret the transaction summary.
    """
    if not genai_enabled:
        return "LLM interpretation disabled (API key missing or invalid)."

    try:
        # Use a model compatible with the API key and desired features
        # 'gemini-1.5-flash-latest' is generally faster and cheaper than 'gemini-pro'
        model = genai.GenerativeModel("gemini-1.5-flash-latest") # Or "gemini-pro"

        prompt_text = f"""
You are a blockchain analyst who explains transaction traces in a simple manner.
Analyze the following Ethereum transaction summary.
Provide a concise yet insightful explanation about what's happening, in plain English.
Focus on the purpose and outcome of the transaction and its internal calls based on the summary provided.

--- Summary ---
{summary}

--- Explanation ---
"""
        # Define generation config if needed (optional)
        generation_config = genai.types.GenerationConfig(
            temperature=0.7,
            max_output_tokens=400
        )

        print(f"[{time.strftime('%X')}] Calling Google AI API (synchronous)...")
        start_time = time.time()
        # Direct synchronous call, remove await and run_in_executor
        response = model.generate_content(
            prompt_text,
            generation_config=generation_config
        )
        end_time = time.time()
        print(f"[{time.strftime('%X')}] Google AI API call finished in {end_time - start_time:.2f} seconds.")

        # Safety feedback check (Optional)
        # if response.prompt_feedback.block_reason:
        #     print(f"LLM call blocked: {response.prompt_feedback.block_reason}")
        #     return f"LLM interpretation blocked due to safety settings: {response.prompt_feedback.block_reason}"

        # Check for empty response parts
        if not response.parts:
            try:
                print(f"Warning: LLM response has no parts. Full response: {response}")
            except Exception:
                 print("Warning: LLM response has no parts and cannot be printed.")

            finish_reason = getattr(response, 'finish_reason', 'UNKNOWN')
            block_reason = getattr(getattr(response, 'prompt_feedback', None), 'block_reason', None)

            if block_reason:
                 return f"LLM interpretation blocked due to safety settings: {block_reason}"
            elif finish_reason != genai.types.generation_types.FinishReason.STOP: # Use enum for comparison
                 return f"LLM interpretation failed or was filtered. Finish reason: {finish_reason.name}"
            else:
                 return "LLM interpretation returned empty content."

        return response.text.strip()

    except Exception as e:
        print(f"Error during Google AI API call: {e}")
        if "API key not valid" in str(e):
             return "LLM interpretation failed: Invalid Google API Key."
        # Consider logging the full exception traceback for debugging
        # import traceback
        # print(traceback.format_exc())
        return f"LLM interpretation failed due to an API error: {type(e).__name__}"


# --- Example: Actual fetch_trace function (Synchronous) ---
def fetch_trace(tx_hash: str) -> Dict[str, Any]: # <-- Remove async
    payload = {
        "jsonrpc": "2.0",
        "method": "debug_traceTransaction",
        "params": [
            tx_hash,
            {"tracer": "callTracer", "tracerConfig": {"withLog": False}},
        ],
        "id": 1,
    }
    REQUEST_TIMEOUT = 180.0
    try:
        # Use synchronous httpx.Client
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client: # <-- Use Client, remove async
            print(f"[{time.strftime('%X')}] Sending debug_traceTransaction request to {NETHERMIND_RPC_URL} for {tx_hash} (timeout={REQUEST_TIMEOUT}s)...")
            start_time = time.time()
            # Direct synchronous call, remove await
            response = client.post(NETHERMIND_RPC_URL, json=payload)
            end_time = time.time()
            print(f"[{time.strftime('%X')}] Received response from RPC node in {end_time - start_time:.2f} seconds. Status: {response.status_code}")

            response.raise_for_status() # Still useful

            result_json = response.json()

            if "error" in result_json:
                print(f"Error from Nethermind RPC for {tx_hash}: {result_json['error']}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Nethermind RPC Error: {result_json['error']['message']} (Code: {result_json['error'].get('code')})",
                )
            if "result" not in result_json:
                print(f"Invalid response from Nethermind for {tx_hash}: 'result' field missing. Response: {result_json}")
                raise HTTPException(
                    status_code=502,
                    detail="Invalid response from Nethermind: 'result' field missing.",
                )
            if result_json["result"] is None:
                print(f"Trace result is null for transaction {tx_hash}.")
                raise HTTPException(
                    status_code=404,
                    detail=f"Trace result not available or null for transaction {tx_hash}.",
                )
            print(f"[{time.strftime('%X')}] Successfully fetched trace for {tx_hash}.")
            return result_json["result"]

    except httpx.TimeoutException:
        print(f"Error: Request to Nethermind RPC timed out after {REQUEST_TIMEOUT} seconds for {tx_hash}.")
        raise HTTPException(
            status_code=504,
            detail=f"Request to Nethermind RPC timed out after {REQUEST_TIMEOUT} seconds.",
        )
    except httpx.RequestError as exc:
        print(f"Error connecting to Nethermind node at {NETHERMIND_RPC_URL} for {tx_hash}. Error: {exc}")
        raise HTTPException(
            status_code=503,
            detail=f"Could not connect to Nethermind node: {exc}",
        )
    except httpx.HTTPStatusError as exc:
        print(f"Nethermind RPC returned error status {exc.response.status_code} for {tx_hash}. Response: {exc.response.text[:500]}")
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"Nethermind RPC request failed with status {exc.response.status_code}.",
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        print(f"Unexpected error during fetch_trace for {tx_hash}: {type(e).__name__} - {e}")
        # import traceback
        # print(traceback.format_exc()) # For detailed debugging
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected internal error fetching trace: {type(e).__name__}",
        )


# --- Summaries and Diagram Builders ---
# [These functions were already synchronous, no changes needed here]
# format_call_trace, get_short_address, generate_mermaid_sequence, create_mermaid_diagram
# ... (코드 생략 - 이전과 동일) ...
def format_call_trace(trace_data: Dict[str, Any], indent: str = "") -> str:
    summary = ""
    try:
        from_addr = trace_data.get("from", "N/A")
        to_addr = trace_data.get("to", "[Contract Creation]")
        input_data = trace_data.get("input", "0x")
        function_sig = input_data[:10] if len(input_data) >= 10 else input_data
        gas_used_hex = trace_data.get("gasUsed", "0x0")
        gas_used_int = int(gas_used_hex, 16) if gas_used_hex and gas_used_hex != "0x" else 0 # Handle "0x" case
        call_type = trace_data.get("type", "UNKNOWN")
        error = trace_data.get("error")
        value_hex = trace_data.get("value", "0x0")
        value_int = int(value_hex, 16) if value_hex and value_hex != "0x" else 0 # Handle "0x" case

        # Handle potential contract creation where 'to' is null but should be derived
        created_addr_info = ""
        if not to_addr and call_type in ["CREATE", "CREATE2"]:
            # For CREATE/CREATE2, the address is in 'output' on success
            output_addr = trace_data.get("output")
            if not error and output_addr and len(output_addr) >= 42 and output_addr.startswith("0x"): # Added startswith check
                to_addr = output_addr # Actual address
                created_addr_info = f"[New Contract: {get_short_address(output_addr)}]"
            elif error:
                 to_addr = "[Contract Creation Failed]"
            else:
                 to_addr = "[Contract Creation Attempt]" # Output might be missing/invalid even without explicit error sometimes
        elif not to_addr:
            to_addr = "[Unknown Target]" # Default if 'to' is null and not CREATE

        target_display = created_addr_info if created_addr_info else get_short_address(to_addr)
        summary += f"{indent}- {call_type} from {get_short_address(from_addr)} to {target_display}"
        if value_int > 0:
            summary += f" | Value: {value_int} wei ({value_hex})"
        summary += "\n"
        summary += f"{indent}  Input (func sig): {function_sig}\n"
        summary += f"{indent}  Gas Used: {gas_used_int} ({gas_used_hex})\n"

        if error:
            summary += f"{indent}  Error: {error}\n"
        else:
            output = trace_data.get("output", "0x")
            # Limit output length for readability
            output_display = (output[:66] + "...") if len(output) > 66 else output
            # Avoid showing the created contract address again in output if already shown above
            if call_type not in ["CREATE", "CREATE2"] or not created_addr_info:
                summary += f"{indent}  Output: {output_display}\n"

        if "calls" in trace_data and trace_data.get("calls"): # Added get for safety
            for sub_call in trace_data["calls"]:
                summary += format_call_trace(sub_call, indent + "  ")

    except Exception as e:
        print(f"Error during format_call_trace step: {e}") # Log error
        summary += f"{indent}  [Error formatting this part of the trace: {e}]\n"

    return summary


def get_short_address(address: Optional[str]) -> str:
    if not address or not isinstance(address, str):
        return "[Invalid Address]"
    # Handle bracketed annotations like [Contract Creation]
    if address.startswith("[") and address.endswith("]"):
        return address
    if len(address) >= 42 and address.startswith("0x"):
        return f"{address[:6]}...{address[-4:]}"
    # Handle shorter strings or non-addresses gracefully
    return str(address)

def generate_mermaid_sequence(
    call_data: Dict[str, Any], alias_map: Dict[str, str], lines: List[str]
):
    try:
        from_addr = call_data.get("from")
        to_addr = call_data.get("to")
        call_type = call_data.get("type", "CALL")
        input_data = call_data.get("input", "0x")
        output_data = call_data.get("output", "0x") # Can be contract address for CREATE
        error = call_data.get("error")
        gas_used_hex = call_data.get("gasUsed", "0x0")
        gas_used_int = int(gas_used_hex, 16) if gas_used_hex and gas_used_hex != "0x" else 0
        value_hex = call_data.get("value", "0x0")
        value_int = int(value_hex, 16) if value_hex and value_hex != "0x" else 0

        # Get aliases safely, defaulting to short address or placeholder
        from_alias = alias_map.get(from_addr, get_short_address(from_addr)) if from_addr else "Unknown"

        # Determine target alias, handling contract creation carefully
        to_alias = "[Unknown Target]"
        target_note_addr = None # Store address for notes if needed

        if call_type in ["CREATE", "CREATE2"]:
            created_addr = None
            # Use the actual created address from 'output' if successful and valid
            if not error and output_data and len(output_data) >= 42 and output_data.startswith("0x"):
                created_addr = output_data
                target_note_addr = created_addr # Address for note
                # Use map if already seen, otherwise generate temp descriptive alias
                to_alias = alias_map.get(created_addr, f"NewContract_{get_short_address(created_addr)}")
            elif error:
                # Use a placeholder key based on creator+input for failed creations if needed for mapping
                placeholder_key = f"CREATE_FAILED_{from_addr}_{input_data[:10]}"
                to_alias = alias_map.get(placeholder_key, "FailedCreate")
            else: # Creation attempt, but output is not a valid address
                placeholder_key = f"CREATE_ATTEMPT_{from_addr}_{input_data[:10]}"
                to_alias = alias_map.get(placeholder_key, "CreateAttempt")

        elif to_addr:
            target_note_addr = to_addr # Address for note
            to_alias = alias_map.get(to_addr, get_short_address(to_addr)) # Use map or short address

        # Construct label
        label = call_type
        if input_data and input_data != "0x":
            input_display = (input_data[:10] + ".." if len(input_data) > 10 else input_data)
            label += f"({input_display})"
        if value_int > 0:
            label += f" [{value_int} wei]"
        # Sanitize label for Mermaid (replace colon which is a syntax element)
        label = label.replace(':', ';')

        # Add sequence lines
        lines.append(f"    {from_alias} ->> {to_alias}: {label}")
        lines.append(f"    activate {to_alias}")
        lines.append(f"    Note right of {to_alias}: Gas: {gas_used_int}")
        if target_note_addr and call_type in ["CREATE", "CREATE2"]: # Show created addr
             lines.append(f"    Note right of {to_alias}: Addr: {get_short_address(target_note_addr)}")

        if "calls" in call_data and call_data.get("calls"): # Use get for safety
            for sub_call in call_data["calls"]:
                if isinstance(sub_call, dict): # Check if sub_call is a dict
                    generate_mermaid_sequence(sub_call, alias_map, lines) # Recursive call

        # Add return/error lines
        if error:
            # Sanitize error message for Mermaid and limit length
            safe_error = str(error).replace(':', ';').replace('\n', ' ')
            error_note = f"Error: {safe_error[:50]}{'...' if len(safe_error) > 50 else ''}"
            lines.append(f"    Note right of {to_alias}: {error_note}")
            # Use different arrow for failure
            fail_msg = f"Failed: {safe_error[:30]}{'...' if len(safe_error) > 30 else ''}"
            lines.append(f"    {to_alias} --x {from_alias}: {fail_msg}")
        else:
            output_display = (output_data[:10] + "..." if len(output_data) > 10 else output_data)
            # Don't show contract address again if it was already shown in creation note
            success_msg = "Success"
            if call_type not in ["CREATE", "CREATE2"]:
                 # Avoid showing large outputs unless necessary
                 if output_data != "0x" and len(output_data) < 66:
                     success_msg += f" ({output_display})"
                 elif len(output_data) >= 66 :
                     success_msg += " (output hidden)"
            elif target_note_addr: # Successful creation
                 success_msg = "Success (Contract Created)"

            success_msg = success_msg.replace(':', ';') # Sanitize
            lines.append(f"    {to_alias} -->> {from_alias}: {success_msg}")

        lines.append(f"    deactivate {to_alias}")

    except Exception as e:
        print(f"Error during generate_mermaid_sequence step: {e}")
        # Add error note in diagram if something goes wrong during generation
        from_alias = alias_map.get(call_data.get("from"), "Unknown")
        to_alias = alias_map.get(call_data.get("to"), "Unknown") # Use get safely
        # Ensure aliases are valid Mermaid identifiers (basic check)
        from_alias = from_alias if from_alias and from_alias[0].isalnum() else "UnknownFrom"
        to_alias = to_alias if to_alias and to_alias[0].isalnum() else "UnknownTo"
        lines.append(f"    Note over {from_alias},{to_alias}: Diagram generation error: {e}")


def create_mermaid_diagram(trace_data: Dict[str, Any]) -> Optional[str]:
    if not trace_data or not isinstance(trace_data, dict):
        print("Warning: Invalid or empty trace_data passed to create_mermaid_diagram.")
        return "sequenceDiagram\n    Note over A,B: Input trace data is invalid or empty"

    participants = set() # Stores actual addresses or placeholder keys
    alias_map = {} # Maps address/key to alias (EOA, ContractA, NewContractB, etc.)
    participant_aliases = set() # Track used aliases like "ContractA"
    contract_counter = 0 # For ContractA, ContractB...
    new_contract_counter = 0 # For NewContractA, NewContractB...

    def assign_alias(addr_or_key: str, base_name: str):
        nonlocal contract_counter, new_contract_counter
        if not isinstance(addr_or_key, str) or addr_or_key in alias_map:
            return # Invalid input or already has an alias

        # Heuristic: Assume the very first 'from' address is the EOA
        is_first_from = not alias_map and 'from' in trace_data and trace_data['from'] == addr_or_key
        current_base = "EOA" if is_first_from else base_name

        if is_first_from:
            alias = "EOA"
            if alias not in participant_aliases:
                 alias_map[addr_or_key] = alias
                 participant_aliases.add(alias)
                 participants.add(addr_or_key)
                 return
            # If EOA alias somehow already exists, fall through to regular naming

        # Generate unique alias (ContractA, ContractB, NewContractA...)
        counter = 0
        if current_base == "Contract":
             counter = contract_counter
        elif current_base == "NewContract":
             counter = new_contract_counter
        # Add more base names if needed

        while True:
            alias = f"{current_base}{chr(ord('A') + counter)}"
            if alias not in participant_aliases:
                alias_map[addr_or_key] = alias
                participant_aliases.add(alias)
                participants.add(addr_or_key)
                # Increment the correct counter
                if current_base == "Contract":
                    contract_counter += 1
                elif current_base == "NewContract":
                    new_contract_counter += 1
                break
            counter += 1
            # Safety break if too many participants (e.g., > 26*N)
            if counter > 52:
                 print(f"Warning: Exceeded participant counter limit for base '{current_base}'. Assigning fallback alias.")
                 fallback_alias = f"{current_base}_fallback_{len(participant_aliases)}"
                 if fallback_alias not in participant_aliases:
                     alias_map[addr_or_key] = fallback_alias
                     participant_aliases.add(fallback_alias)
                     participants.add(addr_or_key)
                 else:
                      print(f"Error: Could not assign fallback alias for {addr_or_key}")
                 break


    def find_participants(call):
        if not isinstance(call, dict): return # Safety check

        from_addr = call.get("from")
        to_addr = call.get("to")
        call_type = call.get("type")
        output_data = call.get("output") # Needed for contract creation addresses
        error = call.get("error")
        input_data = call.get("input","") # For placeholder keys

        # Process 'from' address
        if from_addr and isinstance(from_addr, str) and from_addr not in participants:
             assign_alias(from_addr, "Contract") # Assign ContractX or EOA

        # Process 'to' address or creation target
        target_key = None
        base_alias_name = "Contract"

        if call_type in ["CREATE", "CREATE2"]:
            base_alias_name = "NewContract"
            # Successful creation with valid address output
            if not error and output_data and isinstance(output_data, str) and len(output_data) >= 42 and output_data.startswith("0x"):
                target_key = output_data # Use the actual created address
            else:
                # Failed or ambiguous creation - use a placeholder key
                # Make placeholder reasonably unique: type+from+input_prefix
                target_key = f"{call_type}_{from_addr}_{input_data[:16]}"
        elif to_addr and isinstance(to_addr, str): # Check if to_addr is a string
            target_key = to_addr # Use the target address
            base_alias_name = "Contract" # It's an existing contract (or EOA being called)

        # Assign alias to the target if it's new and valid
        if target_key and target_key not in participants:
             assign_alias(target_key, base_alias_name)

        # Recurse into sub-calls
        if "calls" in call and isinstance(call.get("calls"), list):
            for sub_call in call["calls"]:
                 find_participants(sub_call) # Recurse


    try:
        find_participants(trace_data)
    except RecursionError:
         print("Error: Maximum recursion depth exceeded while finding participants. Trace might be too deep.")
         return "sequenceDiagram\n    Note over A,B: Participant generation failed due to deep recursion."
    except Exception as e:
        print(f"Error finding participants: {e}")
        return f"sequenceDiagram\n    Note over A,B: Error generating participants: {e}"

    if not participants:
         return "sequenceDiagram\n    Note over A,B: No participants found in trace data."

    mermaid_lines = ["sequenceDiagram"]

    # Define participants in Mermaid, trying to put EOA first
    participant_definitions = []
    eoa_def = None
    other_defs = []

    # Sort aliases for consistent ordering (EOA, ContractA, ContractB, NewContractA...)
    # Custom sort key: EOA first, then by alias name
    def sort_key(item):
        alias = item[1]
        if alias == "EOA":
            return "0" # Sorts first
        else:
            return alias # Sort alphabetically

    sorted_aliases = sorted(alias_map.items(), key=sort_key)

    for addr_or_key, alias in sorted_aliases:
        # Try to get a display address (shortened real address or placeholder info)
        display_addr = "[Unknown]"
        if isinstance(addr_or_key, str):
            if addr_or_key.startswith("0x") and len(addr_or_key) >= 42:
                display_addr = get_short_address(addr_or_key)
            elif addr_or_key.startswith("CREATE"): # It's a placeholder key
                display_addr = f"[{alias} Placeholder]" # Indicate it's not a real address shown
            else: # Other keys or unexpected values
                 display_addr = get_short_address(addr_or_key) # Fallback

        # Ensure alias itself is Mermaid-safe (alphanumeric, no spaces etc.)
        # Although our generation logic aims for this, double check.
        safe_alias = ''.join(c if c.isalnum() else '_' for c in alias)
        if not safe_alias: safe_alias = "InvalidAlias"

        # Sanitize display_addr for Mermaid label (<br/> is allowed)
        safe_display_addr = display_addr.replace(':', ';').replace('"', "'")

        participant_line = f'    participant {safe_alias} as {safe_alias}<br/>({safe_display_addr})'
        if safe_alias == "EOA":
            eoa_def = participant_line
        else:
            other_defs.append(participant_line)

    if eoa_def:
        mermaid_lines.append(eoa_def)
    mermaid_lines.extend(other_defs)


    # Generate sequence using the final alias map
    try:
        generate_mermaid_sequence(trace_data, alias_map, mermaid_lines)
    except RecursionError:
        print("Error: Maximum recursion depth exceeded during Mermaid sequence generation.")
        mermaid_lines.append("    Note over EOA: Sequence generation failed (deep recursion)")
    except Exception as e:
        print(f"Error generating mermaid sequence: {e}")
        first_alias = next(iter(alias_map.values()), "Unknown")
        safe_first_alias = ''.join(c if c.isalnum() else '_' for c in first_alias)
        if not safe_first_alias: safe_first_alias = "UnknownStart"
        mermaid_lines.append(
            f"    Note over {safe_first_alias}: Error generating sequence details: {e}"
        )

    return "\n".join(mermaid_lines)


# --- Endpoint (Synchronous) ---
@app.get("/analyze/transaction/{tx_hash}", response_model=AnalysisResponse)
def analyze_transaction( # <-- Remove async
    tx_hash: str = Path(
        ..., min_length=66, max_length=66, pattern="^0x[a-fA-F0-9]{64}$"
    ),
    include_raw_trace: bool = Query(False, description="Include the full raw trace in the response (can be large)"),
):
    """
    Analyzes an Ethereum transaction synchronously:
    1. Fetches the call trace using debug_traceTransaction (BLOCKING CALL).
    2. Generates a text summary of the trace.
    3. Creates a Mermaid sequence diagram.
    4. (If Google API key is configured) Gets an AI interpretation (BLOCKING CALL).
    WARNING: This endpoint is synchronous and will block the server thread.
             Use with caution, especially under load.
    """
    print(f"[{time.strftime('%X')}] Starting analysis for transaction: {tx_hash} (synchronous)")
    start_analysis_time = time.time()

    try:
        # Synchronous call, remove await
        raw_trace_result = fetch_trace(tx_hash)
    except HTTPException as e:
        print(f"[{time.strftime('%X')}] Fetch trace failed for {tx_hash}: Status {e.status_code}, Detail: {e.detail}")
        raise e
    except Exception as e:
        print(f"[{time.strftime('%X')}] Unexpected error calling fetch_trace for {tx_hash}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error fetching trace: {type(e).__name__}",
        )

    # --- Status logic ---
    status = "Success"
    error_message = raw_trace_result.get("error")
    if error_message:
        status = "Failed"
        print(f"[{time.strftime('%X')}] Transaction {tx_hash} trace indicates failure: {error_message}")
    else:
         print(f"[{time.strftime('%X')}] Transaction {tx_hash} trace indicates success (top-level).")

    # --- Text Summary ---
    summary = f"Trace for {tx_hash} ({status}):\n"
    try:
        print(f"[{time.strftime('%X')}] Generating text summary for {tx_hash}...")
        summary += format_call_trace(raw_trace_result)
        print(f"[{time.strftime('%X')}] Text summary generated.")
    except Exception as e:
        summary += f"\n[Error generating text summary: {e}]"
        print(f"[{time.strftime('%X')}] Error during format_call_trace for {tx_hash}: {e}")

    # --- Mermaid Diagram ---
    mermaid_diagram_str = None
    mermaid_explanation = "Mermaid sequence diagram generation failed or was skipped."
    try:
        print(f"[{time.strftime('%X')}] Generating Mermaid diagram for {tx_hash}...")
        mermaid_diagram_str = create_mermaid_diagram(raw_trace_result)
        if mermaid_diagram_str:
            mermaid_explanation = (
                "Below is a Mermaid sequence diagram visualizing the internal calls. "
                # ... (rest of explanation)
            )
            print(f"[{time.strftime('%X')}] Mermaid diagram generated.")
        else:
             print(f"[{time.strftime('%X')}] Mermaid diagram generation resulted in empty output.")
    except Exception as e:
        mermaid_explanation = f"Mermaid sequence diagram generation failed: {e}"
        print(f"[{time.strftime('%X')}] Error during create_mermaid_diagram for {tx_hash}: {e}")

    # --- LLM Interpretation (Synchronous) ---
    interpretation_text = "LLM interpretation skipped or failed."
    if genai_enabled:
        try:
            print(f"[{time.strftime('%X')}] Requesting LLM interpretation for {tx_hash} (synchronous)...")
            # Synchronous call, remove await
            interpretation_text = get_llm_interpretation(summary)
            print(f"[{time.strftime('%X')}] LLM interpretation received.")
        except Exception as e:
            interpretation_text = f"LLM Interpretation failed: {e}"
            print(f"[{time.strftime('%X')}] Error calling get_llm_interpretation for {tx_hash}: {e}")
    else:
        interpretation_text = "LLM interpretation disabled (API key not configured)."
        print(f"[{time.strftime('%X')}] LLM interpretation skipped (API key not configured).")


    end_analysis_time = time.time()
    print(f"[{time.strftime('%X')}] Analysis complete for {tx_hash}. Total time: {end_analysis_time - start_analysis_time:.2f} seconds.")

    return AnalysisResponse(
        transaction_hash=tx_hash,
        status=status,
        error_message=error_message,
        call_trace_summary=summary.strip(),
        mermaid_explanation=mermaid_explanation,
        mermaid_diagram=mermaid_diagram_str,
        raw_trace=raw_trace_result if include_raw_trace else None,
        interpretation=interpretation_text,
    )


# --- Run server ---
if __name__ == "__main__":
    import uvicorn

    # Running a synchronous FastAPI app
    # Uvicorn will likely run the sync endpoint handler in a thread pool,
    # but the handler itself executes sequentially.
    print("Starting synchronous FastAPI server...")
    # reload=False is generally better for non-development environments
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
