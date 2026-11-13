"""
engine.py

Standalone Car Recommendation Engine for German Used Cars.
Aggregates matching listings into vehicle model summaries with TCO and market diagnostics.
"""

from dataclasses import dataclass, field
import os
import re
from typing import List, Optional
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# 1. CONFIGURATION & DOMAIN HEURISTICS
# ---------------------------------------------------------
ENERGY = {"petrol": 2.11, "diesel": 2.25, "kwh": 0.42}
EV_KWH = {
    "Hatchback": 16,
    "SUV": 20,
    "Station Wagon": 18,
    "Sedan": 17,
    "Van / MPV": 22,
    "Coupe / Convertible": 18,
    "default": 19,
}
PHEV_ENGINE_L100 = 6.5
PHEV_E_SHARE = {True: 0.40, None: 0.25, False: 0.10}
DEPRECIATION_BASE = "fair"
EV_PUBLIC_FACTOR = 0.55 / 0.42

PREMIUM_BRANDS = {
    "audi", "bmw", "mercedes-benz", "porsche", "alfa-romeo", "jaguar",
    "land-rover", "maserati", "aston-martin", "bentley", "ferrari",
    "lamborghini", "cadillac", "infiniti", "volvo", "mini"
}

ECONOMY_BRANDS = {
    "dacia", "skoda", "seat", "ford", "opel", "hyundai", "kia",
    "toyota", "renault", "peugeot", "citroen", "fiat", "mazda",
    "nissan", "mitsubishi", "honda", "suzuki", "smart", "volkswagen"
}

SPACE_CARGO = {
    "low": {
        "Hatchback": 1.0,
        "Sedan": 1.0,
        "Coupe / Convertible": 0.9,
        "SUV": 0.8,
        "Station Wagon": 0.8,
        "Van / MPV": 0.6,
    },
    "medium": {
        "Station Wagon": 1.0,
        "SUV": 0.90,
        "Van / MPV": 0.9,
        "Sedan": 0.75,
        "Hatchback": 0.65,
        "Coupe / Convertible": 0.2,
    },
    "high": {
        "Van / MPV": 1.0,
        "Station Wagon": 1.0,
        "SUV": 0.90,
        "Sedan": 0.35,
        "Hatchback": 0.15,
        "Coupe / Convertible": 0.0,
    },
}

SPACE_PASS = {
    "p2": {"Van / MPV": 0.6, "default": 1.0},
    "p34": {
        "Station Wagon": 1.0,
        "SUV": 1.0,
        "Van / MPV": 1.0,
        "Sedan": 0.80,
        "Hatchback": 0.45,
        "Coupe / Convertible": 0.1,
    },
    "p5": {
        "Van / MPV": 1.0,
        "Station Wagon": 1.0,
        "SUV": 0.95,
        "Sedan": 0.65,
        "Hatchback": 0.25,
        "Coupe / Convertible": 0.0,
    },
}

SPORT_BODY = {
    "Coupe / Convertible": 1.0,
    "Sedan": 0.8,
    "Hatchback": 0.65,
    "Station Wagon": 0.55,
}
KM_PREF = {"typical": (50_000, 125_000), "low": (0, 30_000), "any": None}

RELAX_ORDER = [
    "max_mileage",
    "max_age",
    "brand_tier",
    "sporty_loosen",
    "body_types",
    "budget_floor",
    "transmission",
    "fuels_allowed",
    "budget_plus_10pct",
]

# ---------------------------------------------------------
# 2. DATA CLASS: BUYER PROFILE
# ---------------------------------------------------------
FUELS = ["Petrol", "Diesel", "Hybrid", "Electric", "Gas (LPG/CNG)"]
BODIES = [
    "Hatchback",
    "SUV",
    "Station Wagon",
    "Sedan",
    "Van / MPV",
    "Coupe / Convertible",
    "Pickup",
    "Commercial Van",
]


@dataclass
class BuyerProfile:
    budget_max: float
    budget_floor_pct: float = 0.35  # Minimum car price as fraction of budget_max (e.g., 0.35 = 35%)
    annual_km: int = 15_000
    ownership_years: int = 4
    passengers: int = 2
    is_family: bool = False
    cargo: str = "medium"  # low | medium | high
    city_share: float = 0.5
    transmission: str = "any"  # any | automatic | manual
    fuels_allowed: Optional[List[str]] = None
    ev_phev_preference: str = "exclude"  # exclude | open | only_ev_phev
    can_charge_home: Optional[bool] = None
    needs_towbar: bool = False
    needs_awd: bool = False
    body_types: Optional[List[str]] = None
    brand_tier: str = "any"  # any | premium | economy
    power_pref: str = "any"  # any | low | medium | high
    sporty: bool = False
    km_pref: str = "typical"
    km_window: Optional[tuple] = None
    max_age: Optional[int] = None
    max_mileage: Optional[int] = None
    weights: dict = field(default_factory=dict)

    def validate(self):
        assert self.cargo in ("low", "medium", "high")
        assert self.power_pref in ("any", "low", "medium", "high")
        assert self.km_pref in ("typical", "low", "any")
        assert self.transmission in ("any", "automatic", "manual")
        assert self.ev_phev_preference in ("exclude", "open", "only_ev_phev")
        assert 0 <= self.city_share <= 1
        assert self.budget_max > 0 and self.passengers >= 1
        assert 0 <= self.budget_floor_pct < 1


def default_weights(p: BuyerProfile) -> dict:
    w = {
        "cost": 2.5,
        "space": 2.0,
        "condition": 2.0,
        "power": 1.0,
        "fuel_fit": 1.0,
        "value": 2.5,
        "sport": 0.0,
    }
    if p.is_family or p.cargo == "high" or p.passengers >= 4:
        w["space"] = 4.0
        w["cost"] = 2.0
    if p.cargo == "low" and p.passengers <= 2 and not p.is_family:
        w["space"] = 0.5
    w["power"] = (
        0.0
        if p.power_pref == "any"
        else (3.5 if p.power_pref == "high" else 1.5)
    )
    if p.power_pref == "high":
        w["cost"] = 1.0
    if p.sporty:
        w.update({
            "sport": 5.5,
            "power": 2.5,
            "cost": 0.3,
            "space": 0.2,
            "fuel_fit": 0.2,
            "value": 2.0,
        })
    w.update(p.weights)
    return w


# ---------------------------------------------------------
# 3. DATA PREPARATION / LOADER
# ---------------------------------------------------------
SIZE_RULES = [
    # --- 1: city cars
    (
        r"^volkswagen (up!|e-up!|fox|lupo)|^seat (mii|arosa)|^skoda citigo|^toyota (aygo|iq)|^citroen c1|^peugeot (107|108)|^renault twingo|"
        r"^fiat (panda|500$|500e|500c|500 )|^hyundai (i10|getz)|^kia picanto|^ford ka|^smart |^opel (karl|adam|agila)|^dacia spring|^suzuki (alto|celerio)|^mini (one|cooper$)|^daihatsu|^honda e$",
        1,
    ),
    # --- 2: small cars + small SUVs & coupe crossovers
    (
        r"^volkswagen (polo|t-cross|taigo)|^ford (fiesta|puma|ecosport|fusion)|^opel (corsa|mokka|crossland|meriva)|^renault (clio|captur|zoe|modus)|"
        r"^seat (ibiza|arona)|^skoda (fabia|kamiq|rapid|scala)|^peugeot (206|207|208|2008)|^citroen (c3|c3 aircross|c4 cactus|ds3)|^toyota (yaris|yaris cross|c-hr)|"
        r"^kia (rio|stonic|venga|soul|xceed)|^hyundai (i20|kona|bayon)|^suzuki (swift|ignis|baleno|vitara|s-cross)|^nissan (micra|juke|note|leaf)|^mazda (2|cx-3)|"
        r"^honda (jazz|hr-v)|^fiat (punto|grande punto|500x|tipo)|^dacia (sandero|duster|logan)|^audi (a1|a2|q2)|^mini |^opel tigra|^alfa romeo mito|^mitsubishi (space star|colt|asx)|^smart",
        2,
    ),
    # --- 3: compact class + compact SUVs / MPVs
    (
        r"^volkswagen (golf|id\.3|jetta|tiguan$|t-roc|beetle|new beetle|scirocco|caddy|eos|bora)|^audi (a3|s3|rs3|q3|tt|tts)|^bmw (1\d\d|2[0-3]\d|x1|x2|i3|z4|ix1)|"
        r"^mercedes-benz (a|b|cla|gla|slk|clc) ?\d*|^opel (astra|zafira$|combo life|grandland|cascada)|^ford (focus|c-max|kuga|b-max|tourneo connect|tourneo courier|mustang mach-e)|"
        r"^seat (leon|ateca)|^skoda (karoq|yeti|enyaq|roomster)|^renault (megane|kadjar|scenic$|kangoo|arkana)|^peugeot (307|308|3008|rifter|partner tepee)|"
        r"^citroen (c4$|c4 picasso$|berlingo|c5 aircross|ds4)|^toyota (auris|corolla|verso|prius)|^kia (ceed|proceed|niro|sportage|ev6|seltos)|"
        r"^hyundai (i30|ioniq|tucson|ix35|ix20|ioniq 5)|^mazda (3|cx-30|cx-5|mx-5|mx-30|5)|^honda (civic|cr-v)|^nissan (qashqai|pulsar)|^mitsubishi eclipse|"
        r"^fiat (doblo|500l|124)|^volvo (v40|xc40|c40)|^dacia dokker|^ssangyong|^jeep (renegade|compass)|^alfa romeo (giulietta|tonale|147)|^cupra|^lancia",
        3,
    ),
    # --- 4: mid-size cars, mid SUVs, big MPVs
    (
        r"^volkswagen (passat|arteon|cc|tiguan allspace|touran|t[456]|multivan|crossblue)|^audi (a4|s4|rs4|a5|s5|rs5|q5|sq5|e-tron)|^bmw (3\d\d|4\d\d|x3|x4|i4|m3|m4)|"
        r"^mercedes-benz (c|clk|glc|glk|cls|eqc|gle coupe) ?\d*|^opel (insignia|vectra|zafira tourer|zafira life|signum|antara)|^ford (mondeo|s-max|galaxy|edge|tourneo|grand c-max|grand tourneo|mustang)|"
        r"^skoda (octavia|kodiaq)|^seat (tarraco|alhambra|exeo|toledo)|^renault (talisman|laguna|espace|grand scenic|koleos|austral)|^peugeot (508|407|5008|expert|traveller)|"
        r"^citroen (c5|spacetourer|grand c4)|^toyota (avensis|rav|proace verso)|^kia (optima|stinger|sorento|carens)|^hyundai (i40|santa fe|h-1|staria)|^mazda (6|cx-60)|"
        r"^nissan (x-trail|ariya|primastar|nv300)|^mitsubishi outlander|^dacia (lodgy|jogger)|^volvo (s60|v60|xc60|s80|v70)|^jaguar (xe|f-pace|e-pace|i-pace|f-type)|^alfa romeo (giulia|stelvio|159)|"
        r"^porsche (macan|boxster|cayman|718)|^land rover (discovery sport|evoque|freelander|range rover evoque)|^lexus (nx|ux|is|rx)|^saab|^maserati (ghibli|levante|grecale)|^subaru",
        4,
    ),
    # --- 5: large
    (
        r"^skoda superb|^audi (a6|s6|rs6|a7|s7|a8|s8|q7|q8|sq7|sq8|r8)|^bmw (5\d\d|6\d\d|7\d\d|8\d\d|x5|x6|x7|m5|ix)|^mercedes-benz (e|s|cl|gl|gls|gle|g|ml|v|viano|sl|amg gt|maybach|eqs|eqe) ?\d*|"
        r"^volkswagen (touareg|sharan|phaeton|amarok)|^volvo (v90|s90|xc90)|^porsche (cayenne|panamera|taycan|911|99\d|997|991)|^land rover (range rover|discovery|defender)|"
        r"^jaguar (xf|xj)|^toyota (land cruiser|alphard)|^maserati quattroporte|^bentley|^ford (explorer|f 150|f-150|ranger)|^jeep (grand cherokee|wrangler|cherokee|gladiator)|^dodge|^cadillac|^chrysler",
        5,
    ),
]
_SIZE = [(re.compile(p), s) for p, s in SIZE_RULES]


def size_for(model):
    m = str(model).lower().strip()
    for pat, s in _SIZE:
        if pat.search(m):
            return s
    return np.nan


def load_data(parquet_path: Optional[str] = None) -> pd.DataFrame:
    """Loads and enriches the pre-scored car dataset using OS-independent relative paths."""
    if parquet_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        parquet_path = os.path.join(base_dir, "cars_scored.parquet")

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(
            f"Dataset not found at '{parquet_path}'. Make sure 'cars_scored.parquet'"
            " is pushed to your GitHub repository."
        )

    cars = pd.read_parquet(parquet_path)
    cars = cars.drop(
        columns=[
            c
            for c in cars.columns
            if c.startswith("c_")
            or c == "Unnamed: 0"
            or c == "price_model_ok"
        ],
        errors="ignore",
    )

    cars["seven_seats"] = (
        cars["offer_description"]
        .fillna("")
        .str.lower()
        .str.contains(r"7[- ]?sitz|7[- ]?seat|siebensitz|7 sitze")
    )

    _txt = (cars["model"] + " " + cars["offer_description"].fillna("")).str.lower()
    SPORT_RE = (
        r"\bgti\b|\bgtd\b|\bgts\b|\bgt-?r\b|type[ -]?r\b|\bnismo\b|\bjcw\b|john"
        r" cooper works|\babarth\b|\bcupra\b|\bquadrifoglio\b|\bsrt\b|\bopc\b|\bn"
        r" performance\b|\bm performance\b|\bm competition\b|\bturbo"
        r" s\b|\bsvr\b|\bgolf r\b|\bgt3\b|\bgt4\b"
    )
    SPORT_MODEL = (
        r"^audi (s\d|rs ?\d|rs q\d|sq\d|tts|tt rs|r8)|^mercedes-benz .* (?:35|43|45|53|63|65)(?: |$)|^bmw m\d|^bmw (?:135|140|235|240|335|340|435|440|m550)|"
        r"^porsche (?:911|718|cayman|boxster|99\d|997|991)|^mazda mx-5|^toyota (?:supra|gr|gt)|^subaru (?:brz|wrx)|^nissan (?:gt-r|370z|350z)|^honda s2000|"
        r"^chevrolet (?:camaro|corvette)|^ford (?:mustang$|mustang gt|focus rs)|^dodge (?:challenger|charger)|^ferrari|^lamborghini|^jaguar f-type|^alfa romeo 4c|^lotus|^alpine|"
        r"^aston martin|^mclaren|^bmw z4|^fiat 124"
    )
    cars["sport_badge"] = (
        _txt.str.contains(SPORT_RE)
        | cars["model"].str.lower().str.contains(SPORT_MODEL)
        | (
            (cars["brand"] == "ford")
            & _txt.str.contains(r"\bst\b(?![- ]?line)")
            & (cars["body_type"] != "SUV")
        )
        | (
            (cars["brand"] == "mini")
            & _txt.str.contains(r"cooper s|john cooper")
        )
        | (
            (cars["brand"] == "skoda")
            & _txt.str.contains(r"octavia")
            & _txt.str.contains(r"\brs\b")
        )
        | (
            (cars["brand"] == "renault")
            & _txt.str.contains(r"\brs\b|renault sport")
        )
    )

    _d = cars["offer_description"].fillna("").str.lower()
    cars["is_phev"] = (cars["fuel_group"] == "Hybrid") & (
        _d.str.contains(
            r"plug-?in|phev|e-hybrid|ehybrid|recharge|\bgte\b|\biv\b|dm-i|rechargeable"
        )
        | (cars["consumption_l100"] < 2.6)
    )

    _size_map = {m: size_for(m) for m in cars["model"].unique()}
    cars["size_class"] = cars["model"].map(_size_map)

    NO_WAGON = re.compile(
        r"^(renault (twingo|zoe|captur)|toyota (aygo|yaris|prius|iq)|hyundai"
        r" (i10|i20|kona)|peugeot (107|108|208|2008)|volkswagen"
        r" (up!|polo|t-cross)|fiat (panda|500)|citroen (c1|c3)|kia"
        r" (picanto|rio)|ford (ka|fiesta|puma)|opel (corsa|adam|karl)|skoda"
        r" citigo|seat mii|smart)"
    )
    bad = (cars["body_type"] == "Station Wagon") & cars["model"].str.lower().str.match(
        NO_WAGON
    )
    cars.loc[bad, "body_type"] = "Hatchback"
    return cars


# ---------------------------------------------------------
# 4. SCORING & FILTERING FUNCTIONS
# ---------------------------------------------------------
def needed_size(p: BuyerProfile) -> int:
    need = 1
    if p.passengers >= 3:
        need = 3
    if p.is_family or p.passengers >= 4:
        need = 4
    if p.cargo == "medium":
        need = max(need, 2)
    if p.cargo == "high":
        need = max(need, 3 if not p.is_family else 4)
    if p.annual_km >= 30_000:
        need = max(need, 3)
    return need


def space_score(c, p: BuyerProfile):
    cargo = c["body_type"].map(SPACE_CARGO[p.cargo]).fillna(0.4)
    tab = SPACE_PASS[
        "p2" if p.passengers <= 2 else "p34" if p.passengers <= 4 else "p5"
    ]
    pas = c["body_type"].map(lambda b: tab.get(b, tab.get("default", 0.4)))
    need = needed_size(p)
    size = c["size_class"].fillna(2.5)
    fit = (1 - 0.55 * (need - size).clip(lower=0)).clip(0, 1)
    if not p.is_family and p.cargo != "high":
        fit = fit - 0.15 * (size - need - 1).clip(lower=0) * p.city_share
    return (0.4 * cargo + 0.25 * pas + 0.35 * fit).clip(0, 1)


def condition_score(c, p: BuyerProfile):
    win = p.km_window or KM_PREF[p.km_pref]
    age_fit = (1 - (c["vehicle_age"] - 8).clip(lower=0) * 0.7 / 8).clip(0.3, 1)
    if win is None:
        return 0.25 * age_fit + 0.75 * 0.5
    lo, hi = win
    km = c["mileage_in_km"]
    below = 1.0 if lo <= 0 else (0.5 + 0.5 * (km / lo)).clip(upper=1.0)
    above = 1 - ((km - hi).clip(lower=0) / 75_000).clip(0, 1)
    return 0.75 * np.minimum(below, above) + 0.25 * age_fit


def sport_score(c, p: BuyerProfile):
    pw = ((c["power_ps"] - 150) / (320 - 150)).clip(0, 1)
    badge = c["sport_badge"].astype(float)
    # Give Coupe/Convertible and high-output Premium brands extra sportiness weight
    body_s = c["body_type"].map(SPORT_BODY).fillna(0.1)
    return 0.45 * pw + 0.25 * body_s + 0.30 * badge


def power_score(c, p: BuyerProfile):
    ps = c["power_ps"]
    if p.power_pref == "any":
        return pd.Series(0.5, index=c.index)
    if p.power_pref == "high":
        return ((ps - 120) / (280 - 120)).clip(0, 1)
    lo, hi = (0, 105) if p.power_pref == "low" else (105, 175)
    dist = np.where(ps < lo, lo - ps, np.where(ps > hi, ps - hi, 0))
    return pd.Series(1 - np.clip(dist / 60, 0, 1), index=c.index)


def fuel_fit_score(c, p: BuyerProfile):
    s = pd.Series(0.5, index=c.index)
    city, km = p.city_share, p.annual_km
    g = c["fuel_group"]
    if p.ev_phev_preference == "only_ev_phev":
        s[(g == "Electric") | c["is_phev"]] += 0.4
    elif p.ev_phev_preference == "open":
        if city >= 0.6 and p.can_charge_home:
            s[g == "Electric"] += 0.15
            s[g == "Hybrid"] += 0.15
    if city <= 0.4 and km >= 20_000:
        s[g == "Diesel"] += 0.25
    if city >= 0.7 and km < 15_000:
        s[g == "Diesel"] -= 0.3
    if km < 12_000:
        s[g == "Petrol"] += 0.15
    s[g == "Gas (LPG/CNG)"] -= 0.15
    return s.clip(0, 1)


def value_score(c):
    pos = (
        (c["price_in_euro"] - c["fair_price"])
        / (c["price_high"] - c["fair_price"])
    ).clip(0, 1)
    mult = (
        c["price_confidence"]
        .map({"high": 1.0, "medium": 0.7, "low": 0.3})
        .astype(float)
    )
    return 1 - pos * mult


def annual_cost_table(c, p: BuyerProfile):
    km, n = p.annual_km, p.ownership_years
    fuel = c["fuel_cost_per_100km"] * km / 100
    if p.can_charge_home is False:
        fuel = fuel.where(c["fuel_group"] != "Electric", fuel * EV_PUBLIC_FACTOR)
    share = PHEV_E_SHARE[p.can_charge_home] * min(1.0, 20_000 / km)
    kwh = c["body_type"].map(EV_KWH).fillna(EV_KWH["default"])
    fuel_px = np.where(
        c["fuel_type"].isin(["Diesel Hybrid"]), ENERGY["diesel"], ENERGY["petrol"]
    )
    phev = (
        (share * kwh * ENERGY["kwh"] + (1 - share) * PHEV_ENGINE_L100 * fuel_px)
        * km
        / 100
    )
    fuel = fuel.where(~c["is_phev"], pd.Series(phev, index=c.index))
    maint = c["est_annual_maintenance_euro"] * (0.7 + 0.3 * km / 15_000)
    base = (
        c["fair_price"] if DEPRECIATION_BASE == "fair" else c["price_in_euro"]
    )
    depr = base * (1 - (1 - c["depreciation_rate"]) ** n) / n
    out = pd.DataFrame(
        {
            "fuel": fuel,
            "maintenance": maint,
            "insurance": c["est_insurance_euro"],
            "tax": c["est_vehicle_tax_euro"],
            "depreciation": depr,
        },
        index=c.index,
    )
    out["running"] = out[["fuel", "maintenance", "insurance", "tax"]].sum(axis=1)
    out["total"] = out["running"] + out["depreciation"]
    return out


def hard_filters(c, p: BuyerProfile, relaxed):
    budget = p.budget_max * (1.10 if "budget_plus_10pct" in relaxed else 1.0)
    # Fixed floor math: floor_pct is the minimum fraction of budget_max (e.g. 35% of max budget)
    floor_pct = 0.15 if "budget_floor" in relaxed else p.budget_floor_pct
    floor = min(p.budget_max * floor_pct, max(2500.0, p.budget_max - 8000.0))

    steps = [
        (
            "not damaged / export / suspicious / unknown",
            (
                ~c["price_verdict"].isin(
                    ["suspiciously low", "damaged / special condition"]
                )
                & (c["export_flag"] == 0)
                & (c["fuel_group"] != "Other")
                & (c["body_type"] != "Unknown")
            ),
        ),
        (
            f"price between {floor:,.0f} and {budget:,.0f}",
            c["price_in_euro"].between(floor, budget),
        ),
    ]

    # Explicit EV / PHEV Filtering
    if p.ev_phev_preference == "exclude":
        steps.append((
            "exclude Electric & Plug-in Hybrid (PHEV) vehicles",
            (c["fuel_group"] != "Electric") & (~c["is_phev"]),
        ))
    elif p.ev_phev_preference == "only_ev_phev":
        steps.append((
            "only Electric or Plug-in Hybrid (PHEV) vehicles",
            (c["fuel_group"] == "Electric") | c["is_phev"],
        ))
    elif p.can_charge_home is not True and not (
        p.fuels_allowed and "Electric" in p.fuels_allowed
    ):
        steps.append((
            "no pure electric cars without home charging",
            c["fuel_group"] != "Electric",
        ))

    # Brand Tier Filter (e.g. Premium for "impress girls / executive", Economy for "cheap commuter")
    if p.brand_tier == "premium" and "brand_tier" not in relaxed:
        steps.append((
            "premium / prestige brands",
            c["brand"].str.lower().isin(PREMIUM_BRANDS),
        ))
    elif p.brand_tier == "economy" and "brand_tier" not in relaxed:
        steps.append((
            "economy / value brands",
            c["brand"].str.lower().isin(ECONOMY_BRANDS),
        ))

    # Body Type Filter
    if p.body_types and "body_types" not in relaxed:
        steps.append(
            (f"body type in {p.body_types}", c["body_type"].isin(p.body_types))
        )
    else:
        ok = (
            ~c["body_type"].isin(["Commercial Van", "Pickup"])
            if not p.body_types
            else c["body_type"].notna()
        )
        steps.append(("no commercial vans / pickups", ok))

    if p.is_family or p.passengers >= 4:
        steps.append((
            "family / 4+ passengers: no coupe, convertible, or small hatchback",
            ~c["body_type"].isin(["Coupe / Convertible"]),
        ))

    # Strict Size Class Requirement (Blocks NaN sizes when min_size >= 3)
    need = needed_size(p)
    if p.is_family or p.passengers >= 4 or p.cargo == "high":
        min_size = max(3, need if "body_types" in relaxed else need)
        steps.append((
            f"family / cargo size class >= {min_size}",
            c["size_class"].notna() & (c["size_class"] >= (3 if "body_types" in relaxed else min_size)),
        ))
    elif need >= 3:
        steps.append((
            f"interior size class >= {need - 1}",
            c["size_class"].isna() | (c["size_class"] >= (need - 1)),
        ))

    if p.passengers >= 6:
        steps.append(("6+ passengers: listing mentions 7 seats", c["seven_seats"]))
    if p.transmission != "any" and "transmission" not in relaxed:
        steps.append((
            f"{p.transmission} transmission",
            c["transmission_group"].str.lower() == p.transmission,
        ))
    if p.fuels_allowed and "fuels_allowed" not in relaxed:
        steps.append((
            f"fuel in {p.fuels_allowed}",
            c["fuel_group"].isin(p.fuels_allowed),
        ))
    if p.sporty:
        ps_min = 140 if "sporty_loosen" in relaxed else 180
        sport_cond = (
            (c["power_ps"] >= ps_min)
            & c["body_type"].isin(list(SPORT_BODY))
            & (c["sport_badge"] | (c["body_type"] == "Coupe / Convertible"))
        )
        # Prevent eco PHEVs from hijacking sporty searches unless requested
        if p.ev_phev_preference != "only_ev_phev":
            sport_cond = sport_cond & (~c["is_phev"])
        steps.append((
            f"sporty: >= {ps_min} PS, performance model or coupe/convertible",
            sport_cond,
        ))
    if p.power_pref == "high":
        steps.append(
            ("high power preference: at least 150 PS", c["power_ps"] >= 150)
        )
    if p.needs_towbar:
        steps.append(("towbar mentioned in listing", c["has_towbar"] == 1))
    if p.needs_awd:
        steps.append(("all-wheel drive mentioned in listing", c["has_awd"] == 1))
    if p.max_age is not None and "max_age" not in relaxed:
        steps.append((f"age <= {p.max_age}", c["vehicle_age"] <= p.max_age))
    if p.max_mileage is not None and "max_mileage" not in relaxed:
        steps.append((
            f"mileage <= {p.max_mileage:,}",
            c["mileage_in_km"] <= p.max_mileage,
        ))
    mask, funnel = pd.Series(True, index=c.index), []
    for name, m in steps:
        mask &= m
        funnel.append((name, int(mask.sum())))
    return mask, funnel


# ---------------------------------------------------------
# 5. AGGREGATION & RECOMMENDATION CORE
# ---------------------------------------------------------
def explain_model(r, p: BuyerProfile):
    why, warn = [], []
    if p.sporty and r["s_sport"] >= 0.55:
        why.append(
            f"Dynamic & stylish setup: ~{int(r['typical_power_ps'])} PS"
            f" {r['body_type']}"
            + (
                " with performance trim/badge"
                if r["sport_badge_share"] > 0.3
                else ""
            )
        )
    if (p.is_family or default_weights(p)["space"] >= 2.0) and r["s_space"] >= 0.75:
        why.append(
            f"Family-ready {r['body_type']} layout with generous cabin and trunk capacity"
        )
    if r["s_power"] >= 0.75 and p.power_pref != "any":
        why.append(
            f"Strong engine output (~{int(r['typical_power_ps'])} PS) matches your"
            f" {p.power_pref}-power preference"
        )
    if r["s_cost"] >= 0.65 and not p.sporty:
        why.append(
            f"Low ownership cost: ~EUR {r['cost_total']:,.0f}/year"
            f" all-in at {p.annual_km:,} km/yr"
        )
    if r["s_fuel_fit"] >= 0.65:
        why.append(
            f"{r['fuel_group']} (~{int(r['typical_power_ps'])} PS) suits your driving profile"
            f" ({int(p.city_share * 100)}% city)"
        )
    why.append(
        f"High market availability: {int(r['listings_count'])} matching listings analyzed"
    )

    if p.needs_towbar or p.needs_awd:
        warn.append(
            "Towbar / AWD availability depends on specific trim — verify on individual listings"
        )
    if p.passengers >= 6:
        warn.append(
            "7-seat configuration must be confirmed on individual vehicle listings"
        )
    if r["fuel_group"] == "Electric" and p.can_charge_home is not True:
        warn.append(
            "Electric vehicle without home charging: public charging rates apply"
        )
    if r["is_phev_share"] > 0.3:
        warn.append(
            "Frequently listed as Plug-in Hybrid: requires regular charging for best economy"
        )
    elif r["fuel_group"] == "Hybrid":
        warn.append(
            "Self-charging Hybrid: efficient in city traffic, slightly higher consumption on fast Autobahn runs"
        )
    if r["fuel_group"] == "Gas (LPG/CNG)":
        warn.append("LPG/CNG price in cost estimate is a rough benchmark")

    return {"why": why[:4], "warn": warn}


def aggregate_model_recommendations(
    scored_pool,
    buyer_profile: BuyerProfile,
    k: int = 5,
    min_listings: int = 3,
    max_per_brand: int = 2,
):
    if scored_pool.empty:
        return pd.DataFrame()

    def mode_or_first(s):
        m = s.mode()
        return m.iloc[0] if not m.empty else s.iloc[0]

    grouped = scored_pool.groupby(["brand", "model"])
    models_df = grouped.agg(
        listings_count=("price_in_euro", "count"),
        typical_price=("price_in_euro", "median"),
        price_q25=("price_in_euro", lambda x: np.percentile(x, 25)),
        price_q75=("price_in_euro", lambda x: np.percentile(x, 75)),
        fair_price=("fair_price", "median"),
        price_low=("price_low", "median"),
        price_high=("price_high", "median"),
        typical_year=("year", lambda x: int(round(x.median()))),
        typical_km=("mileage_in_km", lambda x: int(round(x.median()))),
        typical_power_ps=("power_ps", lambda x: int(round(x.median()))),
        body_type=("body_type", mode_or_first),
        fuel_group=("fuel_group", mode_or_first),
        transmission_group=("transmission_group", mode_or_first),
        cost_total=("cost_total", "median"),
        cost_fuel=("cost_fuel", "median"),
        cost_maintenance=("cost_maintenance", "median"),
        cost_insurance=("cost_insurance", "median"),
        cost_tax=("cost_tax", "median"),
        cost_depreciation=("cost_depreciation", "median"),
        fit_score=("fit_score", "mean"),
        s_cost=("s_cost", "mean"),
        s_space=("s_space", "mean"),
        s_condition=("s_condition", "mean"),
        s_sport=("s_sport", "mean"),
        s_power=("s_power", "mean"),
        s_fuel_fit=("s_fuel_fit", "mean"),
        sport_badge_share=("sport_badge", "mean"),
        is_phev_share=("is_phev", "mean"),
    ).reset_index()

    models_df = models_df[models_df["listings_count"] >= min_listings]
    models_df = models_df.sort_values("fit_score", ascending=False)
    models_df = models_df[models_df.groupby("brand").cumcount() < max_per_brand]

    top_models = models_df.head(k).copy()
    top_models["reasons"] = top_models.apply(
        lambda r: explain_model(r, buyer_profile), axis=1
    )
    return top_models


def recommend(
    p: BuyerProfile,
    data: pd.DataFrame,
    k: int = 5,
    min_pool: int = 25,
    min_listings: int = 3,
    max_per_brand: int = 2,
):
    p.validate()
    relaxed = []
    mask, funnel = hard_filters(data, p, relaxed)
    applicable = [
        r
        for r in RELAX_ORDER
        if (r == "max_mileage" and p.max_mileage is not None)
        or (r == "max_age" and p.max_age is not None)
        or (r == "brand_tier" and p.brand_tier != "any")
        or (r == "sporty_loosen" and p.sporty)
        or (r == "body_types" and p.body_types)
        or (r == "transmission" and p.transmission != "any")
        or (r == "fuels_allowed" and p.fuels_allowed)
        or r in ("budget_floor", "budget_plus_10pct")
    ]
    for r in applicable:
        if mask.sum() >= min_pool:
            break
        relaxed.append(r)
        mask, funnel = hard_filters(data, p, relaxed)

    pool = data[mask].copy()
    info = {"funnel": funnel, "pool_size": len(pool), "relaxed": relaxed}
    if len(pool) == 0:
        return pool, info

    costs = annual_cost_table(pool, p)
    pool = pool.join(costs.add_prefix("cost_"))
    S = pd.DataFrame(index=pool.index)
    S["cost"] = 1 - pool["cost_total"].rank(pct=True)
    S["space"] = space_score(pool, p)
    S["condition"] = condition_score(pool, p)
    S["sport"] = sport_score(pool, p)
    S["power"] = power_score(pool, p)
    S["fuel_fit"] = fuel_fit_score(pool, p)
    S["value"] = value_score(pool)
    w = default_weights(p)
    wsum = sum(w.values())
    pool["fit_score"] = sum(S[c] * w[c] for c in w) / wsum * 100
    for c in S:
        pool["s_" + c] = S[c]
    pool = pool.sort_values("fit_score", ascending=False)
    info["scored_pool"] = pool
    info["weights"] = w

    model_recs = aggregate_model_recommendations(
        scored_pool=pool,
        buyer_profile=p,
        k=k,
        min_listings=min_listings,
        max_per_brand=max_per_brand,
    )
    return model_recs, info