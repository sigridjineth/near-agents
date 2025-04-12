# main.py
import json
import requests
import logging
import os

import uvicorn
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Define where to store the target price file
# Store it in the same directory as the script for simplicity
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_FILE = os.path.join(SCRIPT_DIR, "target_price.json")
# Ensure the directory exists if needed (though os.path.join usually handles this)
# os.makedirs(SCRIPT_DIR, exist_ok=True) # Usually not needed for script dir

# --- Constants ---
# System prompt isn't directly used in the FastAPI response generation
# but kept here for context if you integrate a real LLM later.
SYSTEM_PROMPT = """
You are a NEAR price alert assistant.
You can set a target price in USD with a direction ('above' or 'below'),
and notify the user when the current NEAR price meets that condition.
If the user hasn't set a target yet, request one.
"""
MODEL_NAME = "llama-v3p1-405b-instruct"  # Placeholder if using real LLM


# --- Pydantic Models for Request/Response ---
class UserInput(BaseModel):
    model: str = Field(..., description="llm model.")
    user_message: str = Field(..., description="The message from the user.")
    # You could add history here if needed for more complex conversations
    # history: Optional[List[Dict[str, str]]] = None


class AssistantResponse(BaseModel):
    assistant_message: str = Field(..., description="The response from the assistant.")
    # You could return updated history here
    # updated_history: Optional[List[Dict[str, str]]] = None
    target_info: Optional[Dict[str, Any]] = Field(
        None, description="Current target price info, if any."
    )


# --- Utility Functions (Adapted for FastAPI context) ---


def load_target_info() -> Optional[dict]:
    """
    Loads target price info from TARGET_FILE.
    Returns None if file not found, empty, or invalid JSON.
    """
    if not os.path.exists(TARGET_FILE):
        logger.warning(f"{TARGET_FILE} not found.")
        return None
    try:
        with open(TARGET_FILE, "r") as f:
            content = f.read().strip()
            if not content:
                logger.info(f"{TARGET_FILE} is empty.")
                return None
            return json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse {TARGET_FILE}: {e}")
        # In a real app, you might want to handle this more gracefully,
        # maybe backup/delete the corrupted file.
        # For now, treat as no target set.
        return None
    except Exception as e:
        logger.error(f"Unexpected error reading {TARGET_FILE}: {e}")
        return None


def save_target_info(data: Optional[dict]) -> None:
    """
    Saves target price info to TARGET_FILE.
    Writes empty string if data is None (clears target).
    """
    try:
        content = ""
        if data is not None:
            content = json.dumps(data)
        with open(TARGET_FILE, "w") as f:
            f.write(content)
        logger.info(
            f"Saved target info to {TARGET_FILE}: {content if content else 'cleared'}"
        )
    except Exception as e:
        logger.error(f"Failed to write {TARGET_FILE}: {e}")
        # Decide how to handle write errors - raise exception? Log and continue?
        # For now, just log it. The application state might become inconsistent.
        # raise HTTPException(status_code=500, detail=f"Failed to save target info: {e}")


def get_near_price() -> Optional[float]:
    """
    Gets current NEAR price (USD) from CoinGecko.
    Returns float or None on error. Logs errors.
    """
    error_message = None
    try:
        url = "https://api.coingecko.com/api/v3/simple/price?ids=near&vs_currencies=usd"
        # In production, remove verify=False or configure certs properly
        response = requests.get(url, timeout=10, verify=False)
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        if "near" in data and "usd" in data["near"]:
            price = float(data["near"]["usd"])
            logger.info(f"Fetched NEAR price: ${price:.4f}")
            return price
        else:
            error_message = "[WARN] CoinGecko response missing 'near'/'usd' fields."
            logger.warning(error_message)
            return None
    except requests.exceptions.Timeout:
        error_message = "[ERROR] Request to CoinGecko timed out."
        logger.error(error_message)
        return None
    except requests.exceptions.RequestException as e:
        error_message = f"[ERROR] RequestException from CoinGecko: {e}"
        logger.error(error_message)
        return None
    except (
        json.JSONDecodeError,
        KeyError,
        ValueError,
    ) as e:  # Added ValueError for float conversion
        error_message = (
            f"[ERROR] Parsing CoinGecko response failed or invalid data: {e}"
        )
        logger.error(error_message)
        return None
    # Ensure error_message is captured if needed outside, though returning None is the signal


def parse_target_message(message: str) -> dict:
    """
    Parses user input like '3.5 above' into {'target': float, 'direction': 'above'|'below'}.
    Raises ValueError on invalid format.
    """
    parts = message.strip().split()
    if len(parts) != 2:
        raise ValueError(
            "Message must be in 'PRICE DIRECTION' format (e.g., '3.5 above' or '$4 below')."
        )

    target_str, direction_str = parts
    direction_str = direction_str.lower()
    if direction_str not in ["above", "below"]:
        raise ValueError("Direction must be 'above' or 'below'.")

    if target_str.startswith("$"):
        target_str = target_str[1:]  # Remove leading '$'

    try:
        target_val = float(target_str)
        if target_val <= 0:
            raise ValueError("Target price must be greater than 0.")
    except ValueError:
        # Catch non-float values
        raise ValueError(f"Invalid price value: '{target_str}'. Must be a number.")

    return {"target": target_val, "direction": direction_str}


# --- FastAPI App ---
app = FastAPI(
    title="NEAR Price Alert Assistant",
    description="Set price targets for NEAR and get alerts.",
)

# --- Main Logic Function (Adapted for FastAPI) ---


def handle_near_price_alert_logic(user_message_content: str) -> str:
    """
    Core logic adapted for a stateless request.
    Reads/writes target file, checks price, generates response string.
    Returns the assistant's message content.
    """
    target_info = load_target_info()
    assistant_explanation = ""
    api_error_message = None  # To capture errors from API calls like get_near_price

    # 1) Handle based on whether a target is already set
    if not target_info:
        # No target set. Try to parse user message as a new target.
        try:
            parsed = parse_target_message(user_message_content)
            save_target_info(parsed)
            # Confirmation message after setting target
            assistant_explanation = (
                f"OK. Target set! I'll let you know when NEAR goes "
                f"{parsed['direction']} ${parsed['target']:.2f}."
            )
            logger.info(f"New target set: {parsed}")
            target_info = parsed  # Update target_info for response model
        except ValueError as e:
            # User input was not a valid target setting command.
            # Assume it's a general query or greeting when no target is set.
            # ** HERE WE RETURN THE HARDCODED RESPONSE FROM YOUR EXAMPLE **
            logger.warning(f"User input parsing failed: {e}. Assuming general query.")
            assistant_explanation = (
                "My apologies for the Korean response earlier! Let me try that again.\n\n"
                "You haven't set a target price alert for NEAR yet. Would you like to set one? "
                "Please respond with a target price in USD and a direction ('above' or 'below'), "
                'e.g. "$3.00 above" or "$2.00 below".'
            )
            # In a real LLM integration, you would call the LLM here with history + system prompt.
            # For example:
            # messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_message_content}]
            # assistant_explanation = call_llm_api(MODEL_NAME, messages) # Replace with actual LLM call
        except Exception as e:
            # Handle unexpected errors during parsing/saving
            logger.error(f"Unexpected error setting target: {e}", exc_info=True)
            assistant_explanation = f"[ERROR] Sorry, an internal error occurred while trying to set the target: {e}"
            # Potentially clear the target file if saving failed midway?
            # save_target_info(None)

    else:
        # Target already exists. Check current price vs target.
        # Also check if user wants to *change* the target
        try:
            # Try parsing message *first* in case user wants to change target
            parsed = parse_target_message(user_message_content)
            save_target_info(parsed)
            assistant_explanation = (
                f"OK. Target updated! I'll let you know when NEAR goes "
                f"{parsed['direction']} ${parsed['target']:.2f}."
            )
            logger.info(f"Target updated: {parsed}")
            target_info = parsed  # Update target_info for response model
            # return assistant_explanation # Exit early after updating

        except ValueError:
            # Input wasn't a new target command, proceed to check price against existing target
            logger.info(
                f"User input '{user_message_content}' not a target command, checking price against existing target: {target_info}"
            )
            current_price = get_near_price()

            if current_price is None:
                # Error fetching price is logged within get_near_price
                # Provide a user-friendly message
                assistant_explanation = (
                    f"Sorry, I couldn't fetch the current NEAR price. "
                    f"Your target is still set for ${target_info['target']:.2f} {target_info['direction']}. "
                    "I'll check again later."
                )
            else:
                t_val = target_info["target"]
                direction = target_info["direction"]
                hit = False
                if direction == "above" and current_price > t_val:
                    hit = True
                elif direction == "below" and current_price < t_val:
                    hit = True

                if hit:
                    # Condition met! Notify and clear target.
                    assistant_explanation = (
                        f"🚨 **ALERT!** NEAR price is now **${current_price:.4f}**, which is {direction} your target of ${t_val:.2f}!\n\n"
                        "Target has been cleared. To set a new one, just tell me the price and direction (e.g., '5.0 above')."
                    )
                    logger.info(
                        f"Target hit! Price ${current_price:.4f} {direction} ${t_val:.2f}. Clearing target."
                    )
                    save_target_info(None)  # Clear the target
                    target_info = None  # Update target_info for response model
                else:
                    # Condition not met. Inform user.
                    assistant_explanation = (
                        f"Current NEAR price is ${current_price:.4f}. "
                        f"Your target (${t_val:.2f} {direction}) hasn't been met yet. "
                        f"I'll keep watching!"
                    )
                    logger.info(
                        f"Target not met. Price ${current_price:.4f}, Target ${t_val:.2f} {direction}."
                    )

        except Exception as e:
            # Handle unexpected errors during price check/logic
            logger.error(
                f"Unexpected error checking price or updating target: {e}",
                exc_info=True,
            )
            assistant_explanation = f"[ERROR] Sorry, an internal error occurred: {e}"

    # This function now directly returns the string content for the assistant's message
    return assistant_explanation, target_info  # Return message and current target state


# --- FastAPI Endpoint ---


@app.post("/chat", response_model=AssistantResponse)
async def chat_endpoint(user_input: UserInput):
    """
    Receives user message, processes the alert logic, and returns the assistant's response.
    """
    logger.info(f"Received user message: '{user_input.user_message}'")

    if not user_input.user_message:
        raise HTTPException(status_code=400, detail="User message cannot be empty.")

    try:
        # Call the core logic function
        assistant_msg_content, current_target_info = handle_near_price_alert_logic(
            user_input.user_message
        )

        # Construct the response
        response = AssistantResponse(
            assistant_message=assistant_msg_content, target_info=current_target_info
        )
        return response

    except HTTPException as http_exc:
        # Re-raise HTTP exceptions (like potential 500 from save_target_info)
        raise http_exc
    except Exception as e:
        # Catch-all for other unexpected errors during processing
        logger.error(f"Unhandled exception in /chat endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"An internal server error occurred: {e}"
        )


# --- Root endpoint for basic check ---
@app.get("/")
async def root():
    return {
        "message": "NEAR Price Alert Assistant API is running. Use the /chat endpoint (POST) to interact."
    }


if __name__ == "__main__":
    # 로컬에서 uvicorn으로 서버 실행
    uvicorn.run(app, host="0.0.0.0", port=8000)

# --- To run the app (using uvicorn) ---
# Save this code as main.py
# Run in terminal: uvicorn main:app --reload --host 0.0.0.0 --port 8000
# (remove --reload in production)

# Example curl request:
# curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"user_message": "hi"}'
# curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"user_message": "3.5 above"}'
# curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"user_message": "check status"}'

# Note: The hardcoded Korean apology response will only be returned if:
# 1. No target_price.json exists or it's empty/invalid.
# 2. The user's message ("hi" in the example) *cannot* be parsed as a valid target command ('price direction').
