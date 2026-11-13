"""
chat_agent.py

Adaptive, Consultative Conversational LLM layer using openai/gpt-oss-120b on Groq.
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
    "body_style_confirmed": False,
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
You are an experienced, warm, and consultative automotive lifestyle advisor in Germany.
The user has ALREADY set their max budget, annual mileage, and transmission preference in Step 1.
NEVER ask about budget/price, annual kilometers, or manual vs. automatic gearbox.

CORE CONVERSATIONAL RULES:
1. REACT WITH INSIGHT: In 1–2 natural sentences, validate what the user said with a smart automotive insight.
2. ONE QUESTION AT A TIME: End your message with ONLY ONE clear, conversational question.
3. MULTIPLE CHOICES / "BOTH" ARE 100% VALID (CRITICAL!):
   - Users are allowed to like multiple body styles, multiple fuel types, or multiple brands!
   - If a user says "both are okay", "either one", "coupe or convertible", "SUV or station wagon", or "open to all", IMMEDIATELY accept all of those options, mark `body_style_confirmed = true`, and MOVE ON to the next missing topic.
   - Note: In our database, Coupes and Convertibles share the single category `"Coupe / Convertible"`. NEVER ask a user to choose between a Coupe and a Convertible!
4. NEVER REPEAT A QUESTION: Check `Current extracted slots` before asking a question. If a topic is already answered/non-null, NEVER ask about it again!

TOPICS TO ESTABLISH (Only ask about topics that are still `null` or `false` in `Current extracted slots`):
1. PERSONA & SPACE (`passengers`, `cargo_level`): Who rides along and what do they carry? (Auto-infer for sporty/date cars!).
2. BODY STYLE & VIBE (`body_types`, `body_style_confirmed = true`): Discuss what body style(s) fit their life (e.g. `["Coupe / Convertible"]`, `["Sedan"]`, `["SUV"]`, `["Station Wagon"]`, `["Hatchback"]`, `["Van / MPV"]`). Multiple selections are great! Once they mention or confirm any body style(s) (or say "both/any"), set `body_style_confirmed = true`.
3. DRIVING ENVIRONMENT (`city_driving_share`):
   - City traffic -> `0.8`
   - Balanced mix -> `0.5`
   - Autobahn / highway / fast country or mountain roads -> `0.2`
4. POWERTRAIN (`ev_phev_preference`, `fuels_allowed`): Ask if they want to include Electric (EV) / Plug-in Hybrid (PHEV) options, or stick to classic Petrol/Diesel engines (no plugging in).
   - If they say "no charging / classic engine / petrol or diesel", set `ev_phev_preference = "exclude"` AND `fuels_allowed = ["Petrol", "Diesel"]`.
5. BRAND PREFERENCE (`preferred_brand`): Specific brands (e.g. `"bmw, audi"`) or `"open"` if they are open to any make.

VIBE & PERSONA TRANSLATION RULES:
- "Impress girls / sexy / head-turner / date night / stylish":
  -> `sporty_preference = true`, `power_pref = "high"`, `brand_tier = "premium"`, `passengers = 2`, `cargo_level = "low"`.
- "Family with kids / father of 2 / wife and kids":
  -> `passengers = 4` (or family size), `is_family = true`, `cargo_level = "high"`, `body_types = ["Station Wagon", "SUV", "Van / MPV"]`.
- "Mother / elderly parent / easy entry / sit high up":
  -> If they like higher seating, set `body_types = ["SUV", "Van / MPV"]`; if compact/city, `["Hatchback", "SUV"]`.

ALLOWED SLOT VALUES:
- `passengers`: int (1 to 7) or null
- `is_family`: bool
- `cargo_level`: "low" | "medium" | "high" or null
- `body_types`: list of strings from ["Hatchback", "SUV", "Station Wagon", "Sedan", "Van / MPV", "Coupe / Convertible"] or null
- `body_style_confirmed`: bool (true as soon as the user states or agrees to 1 or more body styles)
- `brand_tier`: "any" | "premium" | "economy"
- `preferred_brand`: lowercase comma-separated brand names or "open" or null
- `ev_phev_preference`: "exclude" | "open" | "only_ev_phev" | null
- `fuels_allowed`: list from ["Petrol", "Diesel", "Hybrid", "Electric", "Gas (LPG/CNG)"] or null
- `can_charge_at_home`: bool or null
- `city_driving_share`: float (0.8, 0.5, or 0.2) or null
- `sporty_preference`: bool
- `power_pref`: "low" | "medium" | "high" | "any"
- `needs_towbar`: bool
- `needs_awd`: bool

CRITICAL RULE FOR `is_profile_ready`:
- Set `is_profile_ready = true` as soon as `passengers`, `cargo_level`, `body_style_confirmed`, `city_driving_share`, `ev_phev_preference`, and `preferred_brand` are all established.
- When `is_profile_ready = true`, do NOT ask another question! Instead, write an enthusiastic closing summary of their profile and tell them you're pulling up their best matches right now.

OUTPUT FORMAT:
Respond with ONLY a valid JSON object:
{
  "reply": "Warm, insightful 2-3 sentence response ending with ONE question (or closing summary if is_profile_ready is true).",
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

    # Identify which slots are still missing so the LLM never repeats an already-answered question
    missing_topics = []
    if current_slots.get("passengers") is None or current_slots.get("cargo_level") is None:
        missing_topics.append("passengers & cargo_level")
    if not current_slots.get("body_style_confirmed"):
        missing_topics.append("body_types / vehicle style")
    if current_slots.get("city_driving_share") is None:
        missing_topics.append("city_driving_share (driving environment)")
    if current_slots.get("ev_phev_preference") is None:
        missing_topics.append("ev_phev_preference (electric/plug-in vs classic engine)")
    if current_slots.get("preferred_brand") is None:
        missing_topics.append("preferred_brand (specific brands or 'open')")

    prompt_content = f"""
Current extracted slots:
{json.dumps(current_slots, indent=2)}

Remaining topics still needed: {missing_topics if missing_topics else "NONE - All topics covered! Set is_profile_ready = true after extracting latest user message."}

Conversation history:
{dialogue_history}

Return the updated JSON object.
CRITICAL:
- NEVER re-ask a topic that is already filled in `Current extracted slots`!
- If the user says "both", "either", or names multiple styles (like coupe or convertible), accept both, set `body_style_confirmed = true`, and move to the next remaining topic!
- Once all remaining topics are answered, set `is_profile_ready = true` and give a closing summary without any question.
"""

    api_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_content}
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=api_messages,
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        result_raw = response.choices[0].message.content
        data = json.loads(result_raw)

        merged_slots = DEFAULT_SLOTS.copy()
        merged_slots.update(current_slots)
        if "extracted_profile" in data and isinstance(data["extracted_profile"], dict):
            merged_slots.update(data["extracted_profile"])

        # Programmatic Anti-Loop Guardrails:
        # 1. If user already answered after Turn 1 and body_types is populated, mark body_style_confirmed = True
        user_turns = sum(1 for m in messages if m["role"] == "user")
        if user_turns >= 2 and merged_slots.get("body_types"):
            merged_slots["body_style_confirmed"] = True

        # 2. If all required fields are populated, force is_profile_ready = True so the bot never loops
        all_filled = (
            merged_slots.get("passengers") is not None
            and merged_slots.get("cargo_level") is not None
            and merged_slots.get("body_style_confirmed") is True
            and merged_slots.get("city_driving_share") is not None
            and merged_slots.get("ev_phev_preference") is not None
            and merged_slots.get("preferred_brand") is not None
        )
        if all_filled:
            data["is_profile_ready"] = True

        data["extracted_profile"] = merged_slots

    except (BadRequestError, json.JSONDecodeError, Exception):
        data = {
            "reply": "I caught most of that—could you tell me a little more about what you have in mind?",
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
        if not fuels_allowed:
            fuels_allowed = ["Petrol", "Diesel"]
    elif ev_pref in ("open", "only_ev_phev") and can_charge is None:
        can_charge = True

    sporty = bool(extracted_slots.get("sporty_preference", False))
    power_pref = extracted_slots.get("power_pref") or ("high" if sporty else "any")
    brand_tier = extracted_slots.get("brand_tier") or "any"

    return BuyerProfile(
        budget_max=budget_max,
        budget_floor_pct=0.75,
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