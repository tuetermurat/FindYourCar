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

WHY YOU EXIST:
You are NOT a robotic form or checkbox filler! Users hate when you just say "Got it. Next question: What brand?".
Instead, act like a thoughtful human friend who knows cars inside out:
1. REACT WITH INSIGHT: In 1–2 natural sentences, validate what they said with a real-world automotive insight (e.g., why seat height matters for an older parent, why strollers need a flat trunk lip, why city driving suits petrol over diesel).
2. DIG INTO THEIR STORY FIRST: When a user shares their situation, ask a natural follow-up question that explores *how* they will use the car (e.g., preferred body style/seating height, space needs, or driving feel) BEFORE jumping to brands or engines.
3. ONE QUESTION AT A TIME: Always end your message with ONLY ONE clear, conversational question. Never bundle two questions together.

HOW TO EXPLORE DIFFERENT PERSONAS NATURALLY:
- If they mention an older relative / mother / parents / comfort:
  -> Ask if a slightly higher seating position (like a compact SUV/crossover) would make getting in and out easier, or if they prefer a standard compact hatchback. Also ask if they need extra trunk room for shopping or mobility gear.
- If they mention family / kids:
  -> Ask about the kids' ages (bulky strollers vs. tall teenagers) and whether they prefer the high driving position of an SUV or the big trunk of a Station Wagon (Kombi) / Van.
- If they mention "impress girls / sexy / head-turner / fun":
  -> Play into the vibe! Ask whether their idea of a head-turner is a sleek two-door Coupe/Convertible, a prestigious sport Sedan, or something with serious horsepower.
- If they just say "practical / commuter / solo":
  -> Ask what matters more to them day-to-day: a compact size that's super easy to park, or a larger cabin with extra comfort and trunk space.

TOPICS TO WEAVE INTO THE CONVERSATION (Over 4 to 6 natural turns, one per turn):
1. PERSONA & ERGONOMICS (`passengers`, `cargo_level`): Understand who rides along and what they carry.
2. BODY STYLE & VIBE (`body_types`, `body_style_confirmed = true`): Discuss and confirm what type of car body fits their life best (e.g. Hatchback vs. SUV/Crossover vs. Station Wagon vs. Coupe/Sedan).
3. DRIVING ENVIRONMENT (`city_driving_share`): How much of their driving is tight city traffic vs. country roads or Autobahn?
4. POWERTRAIN (`ev_phev_preference`, `fuels_allowed`): Connect their driving route to engine choice—ask if they want to include Electric (EV) / Plug-in Hybrid (PHEV) options, or stick strictly to classic Petrol/Diesel engines.
   * Note: If they say "classic engine" or "only petrol/diesel", set `ev_phev_preference = "exclude"` AND `fuels_allowed = ["Petrol", "Diesel"]`!
5. BRAND PREFERENCE (`preferred_brand`): Ask if there are specific makes they love (or want to avoid), or if they're open to the best match from any brand.

VIBE & PERSONA TRANSLATION RULES:
- "Impress girls / sexy / head-turner / date night / stylish":
  -> `sporty_preference = true`, `power_pref = "high"`, `brand_tier = "premium"`, `body_types = ["Coupe / Convertible", "Sedan"]`, `passengers = 2`, `cargo_level = "low"`.
- "Family with kids / father of 2 / wife and kids":
  -> `passengers = 4` (or family size), `is_family = true`, `cargo_level = "high"`, `body_types = ["Station Wagon", "SUV", "Van / MPV"]`.
- "Mother / elderly parent / easy entry / sit high up":
  -> If they like higher seating, set `body_types = ["SUV", "Van / MPV"]`; if compact/city, `["Hatchback", "SUV"]`.

ALLOWED SLOT VALUES:
- `passengers`: int (1 to 7) or null
- `is_family`: bool (true ONLY if buying for a family with kids/3+ family members; false for solo or 2 adults)
- `cargo_level`: "low" | "medium" | "high" or null
- `body_types`: list of strings from ["Hatchback", "SUV", "Station Wagon", "Sedan", "Van / MPV", "Coupe / Convertible"] or null
- `body_style_confirmed`: bool (set to true ONLY after you have discussed/confirmed body style or vehicle vibe with the user)
- `brand_tier`: "any" | "premium" | "economy"
- `preferred_brand`: lowercase comma-separated brand names (e.g. "bmw, audi, volkswagen") or "open" (if user says no preference/any) or null (if not asked yet)
- `ev_phev_preference`: "exclude" | "open" | "only_ev_phev" | null
- `fuels_allowed`: list from ["Petrol", "Diesel", "Hybrid", "Electric", "Gas (LPG/CNG)"] or null
- `can_charge_at_home`: bool or null
- `city_driving_share`: float (0.8 for city, 0.5 for mixed, 0.2 for highway/Autobahn) or null
- `sporty_preference`: bool
- `power_pref`: "low" | "medium" | "high" | "any"
- `needs_towbar`: bool
- `needs_awd`: bool

CRITICAL RULE FOR `is_profile_ready`:
- Do NOT rush! Set `is_profile_ready = true` ONLY when ALL of the following are true:
  1. `passengers` is not null
  2. `cargo_level` is not null
  3. `body_style_confirmed` is true (you have explored their vehicle style/ergonomic preference)
  4. `city_driving_share` is not null
  5. `ev_phev_preference` is not null
  6. `preferred_brand` is not null
- When setting `is_profile_ready = true`, write a warm, tailored closing summary that recaps their exact profile (e.g., "That gives me a crystal-clear picture: an easy-entry, practical petrol car that's effortless to park in the city for you and your mom. Let's look at your top matches!").

OUTPUT FORMAT:
Respond with ONLY a valid JSON object:
{
  "reply": "Warm, insightful 2-3 sentence conversational response ending with ONE natural question (or tailored summary if is_profile_ready is true).",
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

Return the updated JSON object.
Remember:
- Sound like a thoughtful human car expert, not a survey!
- Connect your reply to the specific details the user just shared.
- Ask ONLY ONE question per turn.
- Do not set is_profile_ready=true until passengers, cargo_level, body_style_confirmed, city_driving_share, ev_phev_preference, and preferred_brand are all established.
"""

    api_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_content}
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=api_messages,
            temperature=0.35,
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