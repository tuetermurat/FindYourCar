"""
chat_agent.py

Conversational LLM layer using openai/gpt-oss-120b on Groq.
Extracts lifestyle parameters into engine.BuyerProfile.
Topic 2 is now dedicated to Brand Preference early in the conversation.
"""

import os
import json
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from groq import Groq, BadRequestError
from engine import BuyerProfile

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file.")

client = Groq(api_key=api_key)
MODEL_NAME = "openai/gpt-oss-120b"

GERMAN_DEFAULT_CITY = 0.5

SYSTEM_PROMPT = """
You are a warm, conversational car lifestyle advisor helping a buyer in Germany find their ideal car.
The user has ALREADY entered their budget, transmission preference, and annual mileage in the sidebar. 
NEVER ask about:
- Budget or price limits
- Manual vs automatic transmission
- How many kilometers they drive per year

YOUR CONVERSATION CHECKLIST (Cover these 5 topics in exact sequence, asking ONE question per turn):
1. PASSENGERS & FAMILY: Who rides in the car regularly? (Family, teenagers, kids, pets, solo commute).
2. BRAND PREFERENCE (ASK THIS 2ND): Ask directly and warmly: "Do you have any preferred car brands (for example, Audi, Volkswagen, BMW, Skoda, Toyota), or are you completely open to any make?"
   * Mapping:
     - If they say "no", "any", "no preference", "open to all", or "doesn't matter" -> set preferred_brand = null
     - If they mention specific brand(s) (e.g., "Audi", "BMW and Mercedes", "VW") -> set preferred_brand = "audi" (or comma-separated if multiple, e.g., "audi, volkswagen")
3. LUGGAGE & ACTIVITIES: What items are carried in the trunk? (Holidays, camping, sports gear, strollers, big dog crates, or just grocery bags?).
   * FORBIDDEN: NEVER use words like "low", "medium", or "high". Ask about real items/trips.
   * Mapping: 
     - Only groceries/handbags -> cargo_level = "low"
     - Standard suitcases, sports bags -> cargo_level = "medium"
     - Heavy holiday gear, 4+ suitcases, camping gear, strollers, big dogs, or "a lot of stuff" -> cargo_level = "high"
4. DRIVING ROUTE / ENVIRONMENT: What does their everyday drive look like? (Mostly city stop-and-go traffic, country roads, or high-speed Autobahn / highway driving?).
   * Mapping:
     - Mostly city / short urban errands -> city_driving_share = 0.8
     - Balanced mix (commute + holidays) -> city_driving_share = 0.5
     - Mostly Autobahn / highway / long distances -> city_driving_share = 0.2
5. CHARGING & PARKING: Do they have a private garage, driveway, or workplace charger where an electric/plug-in car could be plugged in, or do they park on the street?

CRITICAL FLOW RULE FOR is_profile_ready:
- You must NOT set `is_profile_ready = true` until ALL 5 topics have been addressed!
- Do not conclude the conversation early.

OUTPUT FORMAT:
Respond with ONLY a valid JSON object starting with '{' and ending with '}'.

{
  "reply": "Warm, natural response acknowledging what they said, followed by your next single question from the sequence.",
  "extracted_profile": {
    "passengers": int or null,
    "cargo_level": "low" | "medium" | "high" or null,
    "city_driving_share": float or null,
    "can_charge_at_home": bool or null (true if private garage/socket, false if street parking or unknown),
    "needs_towbar": bool,
    "needs_awd": bool,
    "sporty_preference": bool,
    "preferred_brand": string or null
  },
  "is_profile_ready": bool (Set to true ONLY AFTER Topic 5 charging/parking has been answered)
}

When `is_profile_ready` is true:
- Set your `reply` to an enthusiastic closing message telling the user you have everything needed and are pulling up their best car matches right now.
"""


def run_chat_turn(
    messages: List[Dict[str, str]],
    current_slots: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Sends conversation history to openai/gpt-oss-120b and extracts updated lifestyle slots.
    """
    if current_slots is None:
        current_slots = {
            "passengers": None,
            "cargo_level": None,
            "city_driving_share": None,
            "can_charge_at_home": None,
            "needs_towbar": False,
            "needs_awd": False,
            "sporty_preference": False,
            "preferred_brand": None
        }

    dialogue_history = ""
    for m in messages:
        sender = "User" if m["role"] == "user" else "Advisor"
        dialogue_history += f"{sender}: {m['content']}\n"

    prompt_content = f"""
Current extracted slots:
{json.dumps(current_slots, indent=2)}

Conversation history:
{dialogue_history}

Generate the updated JSON object according to instructions:
"""

    api_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_content}
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=api_messages,
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        result_raw = response.choices[0].message.content
        data = json.loads(result_raw)
    except (BadRequestError, json.JSONDecodeError, Exception):
        data = {
            "reply": "Understood! I'll match the best cars suited for you and your family.",
            "extracted_profile": current_slots,
            "is_profile_ready": True
        }

    return data


def create_buyer_profile(
    budget_max: float,
    transmission: str,
    annual_km: int,
    extracted_slots: Dict[str, Any]
) -> BuyerProfile:
    """
    Merges zero-token UI form inputs (budget, transmission, annual km)
    with extracted conversational lifestyle parameters.
    """
    return BuyerProfile(
        budget_max=budget_max,
        transmission=transmission,
        annual_km=annual_km,
        budget_floor_pct=0.30,
        passengers=extracted_slots.get("passengers") or 2,
        cargo=extracted_slots.get("cargo_level") or "medium",
        city_share=extracted_slots.get("city_driving_share") if extracted_slots.get("city_driving_share") is not None else GERMAN_DEFAULT_CITY,
        can_charge_home=extracted_slots.get("can_charge_at_home"),
        needs_towbar=bool(extracted_slots.get("needs_towbar", False)),
        needs_awd=bool(extracted_slots.get("needs_awd", False)),
        sporty=bool(extracted_slots.get("sporty_preference", False))
    )