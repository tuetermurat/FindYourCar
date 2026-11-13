"""
chat_agent.py

Adaptive Conversational LLM layer using openai/gpt-oss-120b on Groq.
Extracts rich lifestyle parameters, vehicle vibes, and EV/PHEV preferences into engine.BuyerProfile.
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

BRAND_ALIASES = {
    "vw": "volkswagen",
    "volvo": "volvo",
    "merc": "mercedes-benz",
    "mercedes": "mercedes-benz",
    "benz": "mercedes-benz",
    "alfa": "alfa-romeo",
    "alfa romeo": "alfa-romeo",
    "aston": "aston-martin",
    "aston martin": "aston-martin",
    "land rover": "land-rover",
    "range rover": "land-rover",
    "chevy": "chevrolet",
    "ssang yong": "ssangyong",
}

DEFAULT_SLOTS: Dict[str, Any] = {
    "passengers": None,
    "is_family": False,
    "cargo_level": None,
    "body_types": None,
    "brand_tier": "any",
    "preferred_brand": None,
    "ev_phev_preference": None,
    "fuels_allowed": None,
    "can_charge_at_home": None,
    "city_driving_share": None,
    "sporty_preference": False,
    "power_pref": "any",
    "needs_towbar": False,
    "needs_awd": False,
}

SYSTEM_PROMPT = """
You are a sharp, warm, and perceptive automotive lifestyle advisor in Germany.
The user has ALREADY locked in their max budget, annual mileage, and transmission preference in Step 1.
NEVER ask about budget/price, annual kilometers, or manual vs. automatic gearbox.

STRICT CONVERSATION STYLE RULES (CRITICAL!):
1. ASK ONLY ONE SINGLE QUESTION PER TURN! Never combine two questions in one message.
2. KEEP IT SHORT: Maximum 2 sentences per reply (1 brief sentence acknowledging their answer + 1 clear, easy-to-read question).
3. INFER SMARTLY SO YOU DON'T ASK UNNECESSARY QUESTIONS: When the user describes their vibe or family situation, automatically fill in the implied slots using the Vibe Rules below, and move straight to the next missing piece of information.

THE 4 CORE THINGS TO KNOW BEFORE SETTING `is_profile_ready = true` (Ask ONE at a time, only for what is still `null`):
1. LIFESTYLE / VIBE / WHO IT'S FOR: Who is riding in the car and what is the vibe/cargo need? (If they already told you in their first message, infer `passengers` and `cargo_level` immediately!).
2. BRAND PREFERENCE: Ask if they have any favorite car brands (e.g., BMW, Audi, VW, Skoda, Toyota) or if they are open to all makes.
3. DRIVING ENVIRONMENT: Ask whether their daily routine is mostly city traffic, a balanced mix, or mostly Autobahn/highway driving.
4. EV / PHEV PREFERENCE: Ask directly: "Would you like to include Electric (EV) or Plug-in Hybrid (PHEV) cars, or should we stick strictly to classic Petrol and Diesel engines?"
   * (If they say yes/open to EV/PHEV, you can assume `can_charge_at_home = true` unless they mention street parking. If they say no/classic, set `ev_phev_preference = "exclude"`).

VIBE & PERSONA TRANSLATION RULES (Apply automatically!):
- "Impress girls / sexy / head-turner / date night / stylish / sporty coupe":
  -> `sporty_preference = true`, `power_pref = "high"`, `brand_tier = "premium"`, `body_types = ["Coupe / Convertible", "Sedan"]`, `passengers = 2`, `cargo_level = "low"`. (Do NOT ask about luggage or passengers!).
- "Family with kids / father of 2 / wife and kids":
  -> `passengers = 4` (or total family size), `is_family = true`, `cargo_level = "high"`, `body_types = ["Station Wagon", "SUV", "Van / MPV"]` (unless they explicitly asked for a Sedan).
- "Sit high up / SUV / safe feeling / easy entry":
  -> `body_types = ["SUV"]`, `cargo_level = "medium"` (if not set).
- "Autobahn cruiser / executive / long-distance comfort":
  -> `city_driving_share = 0.2`, `power_pref = "high"`, `body_types = ["Sedan", "Station Wagon", "SUV"]`.
- "Cheap commuter / student / first car / low running costs":
  -> `brand_tier = "economy"`, `power_pref = "low"`, `cargo_level = "low"`, `body_types = ["Hatchback"]`.
- "Mountain / winter / snow / ski trips":
  -> `needs_awd = true`.

ALLOWED SLOT VALUES:
- `passengers`: int (1 to 7) or null
- `is_family`: bool (true if buying for family/kids)
- `cargo_level`: "low" | "medium" | "high" or null
- `body_types`: list of strings from ["Hatchback", "SUV", "Station Wagon", "Sedan", "Van / MPV", "Coupe / Convertible"] or null
- `brand_tier`: "any" | "premium" | "economy"
- `preferred_brand`: lowercase comma-separated brand names (e.g. "bmw, audi, volkswagen") or "open" (if user says no preference/any) or null (if not asked yet)
- `ev_phev_preference`: "exclude" (wants classic petrol/diesel, no EV/PHEV) | "open" (open to EV/PHEV) | "only_ev_phev" (wants only EV or plug-in) | null (if not asked/known yet)
- `fuels_allowed`: list from ["Petrol", "Diesel", "Hybrid", "Electric", "Gas (LPG/CNG)"] or null
- `can_charge_at_home`: bool or null
- `city_driving_share`: float (0.8 for city, 0.5 for mixed, 0.2 for highway/Autobahn) or null
- `sporty_preference`: bool
- `power_pref`: "low" | "medium" | "high" | "any"
- `needs_towbar`: bool
- `needs_awd`: bool

CRITICAL RULE FOR `is_profile_ready`:
- Set `is_profile_ready = true` ONLY when `passengers`, `cargo_level`, `preferred_brand`, `city_driving_share`, and `ev_phev_preference` are ALL non-null.
- If `is_profile_ready` is already true and the user sends a follow-up tweak, update the slots and keep `is_profile_ready = true`.

OUTPUT FORMAT:
Respond with ONLY a valid JSON object:
{
  "reply": "Short, natural 1-2 sentence reply with ONLY ONE question (or closing message if is_profile_ready is true).",
  "extracted_profile": { ...all slots... },
  "is_profile_ready": bool
}
"""


def normalize_brand_list(brand_str: Optional[str]) -> List[str]:
    """Normalizes conversational brand names into exact dataset brand strings."""
    if not brand_str or not isinstance(brand_str, str):
        return []
    clean = brand_str.strip().lower()
    if clean in ("null", "none", "any", "no", "no preference", "open", "all"):
        return []
    raw_parts = [
        b.strip().strip("'\"")
        for b in clean.replace(" and ", ",").replace("/", ",").split(",")
        if b.strip()
    ]
    normalized = []
    for part in raw_parts:
        mapped = BRAND_ALIASES.get(part, part)
        normalized.append(mapped)
    return normalized


def run_chat_turn(
    messages: List[Dict[str, str]],
    current_slots: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Sends conversation history to Groq and updates structured lifestyle slots."""
    if current_slots is None:
        current_slots = DEFAULT_SLOTS.copy()

    dialogue_history = ""
    for m in messages:
        sender = "User" if m["role"] == "user" else "Advisor"
        dialogue_history += f"{sender}: {m['content']}\n"

    prompt_content = f"""
Current extracted slots:
{json.dumps(current_slots, indent=2)}

Conversation history:
{dialogue_history}

Return the updated JSON object. CRITICAL: Ask ONLY ONE short question in your `reply`! Do not bundle multiple questions together.
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

        merged_slots = DEFAULT_SLOTS.copy()
        merged_slots.update(current_slots)
        if "extracted_profile" in data and isinstance(data["extracted_profile"], dict):
            merged_slots.update(data["extracted_profile"])
        data["extracted_profile"] = merged_slots

    except (BadRequestError, json.JSONDecodeError, Exception):
        data = {
            "reply": "I caught most of that—could you rephrase that last part slightly?",
            "extracted_profile": current_slots,
            "is_profile_ready": False
        }

    return data


def create_buyer_profile(
    budget_max: float,
    transmission: str,
    annual_km: int,
    extracted_slots: Dict[str, Any]
) -> BuyerProfile:
    """
    Merges baseline UI parameters with rich conversational lifestyle & vibe parameters.
    """
    passengers = extracted_slots.get("passengers") or 2
    is_family = bool(extracted_slots.get("is_family", False)) or (passengers >= 4)
    cargo = extracted_slots.get("cargo_level") or ("high" if is_family else "medium")

    body_types = extracted_slots.get("body_types")
    if isinstance(body_types, list) and len(body_types) == 0:
        body_types = None

    fuels_allowed = extracted_slots.get("fuels_allowed")
    if isinstance(fuels_allowed, list) and len(fuels_allowed) == 0:
        fuels_allowed = None

    ev_pref = extracted_slots.get("ev_phev_preference") or "exclude"
    can_charge = extracted_slots.get("can_charge_at_home")
    if ev_pref == "exclude":
        can_charge = False
    elif ev_pref in ("open", "only_ev_phev") and can_charge is None:
        can_charge = True

    sporty = bool(extracted_slots.get("sporty_preference", False))
    power_pref = extracted_slots.get("power_pref") or ("high" if sporty else "any")
    brand_tier = extracted_slots.get("brand_tier") or "any"

    return BuyerProfile(
        budget_max=budget_max,
        budget_floor_pct=0.75,  # Look between 75% and 100% of max budget (e.g. €15k-€20k for €20k budget)
        annual_km=annual_km,
        transmission=transmission,
        passengers=int(passengers),
        is_family=is_family,
        cargo=cargo,
        city_share=extracted_slots.get("city_driving_share") if extracted_slots.get("city_driving_share") is not None else GERMAN_DEFAULT_CITY,
        can_charge_home=can_charge,
        ev_phev_preference=ev_pref,
        fuels_allowed=fuels_allowed,
        body_types=body_types,
        brand_tier=brand_tier,
        power_pref=power_pref,
        sporty=sporty,
        needs_towbar=bool(extracted_slots.get("needs_towbar", False)),
        needs_awd=bool(extracted_slots.get("needs_awd", False)),
    )