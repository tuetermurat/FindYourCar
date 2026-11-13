"""
app.py

FindYourCar — AI-Powered Used Car Lifestyle Recommender
Built with Streamlit, Groq (openai/gpt-oss-120b), and custom domain ML engine.
"""

import os
import streamlit as st
from dotenv import load_dotenv
from engine import load_data, recommend
from chat_agent import (
    run_chat_turn,
    create_buyer_profile,
    DEFAULT_SLOTS,
    normalize_brand_list,
)

load_dotenv()

st.set_page_config(
    page_title="FindYourCar — AI Lifestyle Recommender",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# CUSTOM 3D TACTILE CSS STYLING
# ---------------------------------------------------------
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* 1. 3D TACTILE SIDEBAR CARDS & CONTROLS */
    [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff;
        border: 1px solid #d1d5db !important;
        border-radius: 14px !important;
        box-shadow: 0 4px 0 #9ca3af, 0 6px 10px rgba(0, 0, 0, 0.08) !important;
        margin-bottom: 1rem;
        transition: all 0.12s ease-in-out;
    }

    [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 0 #9ca3af, 0 10px 14px rgba(0, 0, 0, 0.12) !important;
    }

    [data-testid="stSidebar"] button {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%) !important;
        border: 1px solid #94a3b8 !important;
        border-radius: 10px !important;
        color: #1e293b !important;
        font-weight: 600 !important;
        padding: 0.6rem 1rem !important;
        box-shadow: 0 4px 0 #94a3b8, 0 5px 8px rgba(0,0,0,0.1) !important;
        transition: all 0.08s ease !important;
    }

    [data-testid="stSidebar"] button:hover {
        background: linear-gradient(180deg, #ffffff 0%, #edf2f7 100%) !important;
        border-color: #64748b !important;
        transform: translateY(-1px);
        box-shadow: 0 5px 0 #64748b, 0 7px 10px rgba(0,0,0,0.12) !important;
    }

    [data-testid="stSidebar"] button:active {
        transform: translateY(4px) !important;
        box-shadow: 0 0 0 #94a3b8, inset 0 2px 4px rgba(0,0,0,0.15) !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label {
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 6px 12px;
        margin-bottom: 6px;
        box-shadow: 0 2px 0 #cbd5e1;
        transition: all 0.1s ease;
        display: flex;
        cursor: pointer;
    }

    /* 2. 3D CHAT MESSAGE BUBBLES */
    [data-testid="stChatMessage"] {
        background-color: transparent !important;
        padding: 0.6rem 0.2rem !important;
        margin-bottom: 0.8rem;
    }

    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]),
    [data-testid="stChatMessage"]:nth-child(odd) {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 14px !important;
        box-shadow: 0 4px 0 #cbd5e1, 0 6px 12px rgba(0, 0, 0, 0.05) !important;
        padding: 1rem 1.25rem !important;
    }

    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]),
    [data-testid="stChatMessage"]:nth-child(even) {
        background: #f8fafc !important;
        border: 1px solid #94a3b8 !important;
        border-radius: 14px !important;
        box-shadow: 0 4px 0 #64748b, 0 6px 12px rgba(0, 0, 0, 0.06) !important;
        padding: 1rem 1.25rem !important;
    }

    /* 3. 3D CHAT INPUT BAR */
    [data-testid="stChatInput"] {
        border-radius: 14px !important;
        box-shadow: 0 5px 0 #94a3b8, 0 8px 16px rgba(0, 0, 0, 0.1) !important;
        border: 1px solid #cbd5e1 !important;
        background: #ffffff !important;
    }

    /* 4. HERO BANNER & METRICS */
    [data-testid="stMetricValue"] {
        font-size: 1.45rem !important;
        font-weight: 700;
    }
    
    .hero-card {
        padding: 1.4rem 1.6rem;
        border-radius: 14px;
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 5px 0 #0f172a, 0 8px 16px rgba(0, 0, 0, 0.2);
    }
    .hero-card h1 {
        color: #f8fafc;
        margin-bottom: 0.25rem;
        font-size: 2.1rem;
    }
    .hero-card p {
        color: #94a3b8;
        font-size: 1rem;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_cars_data():
    return load_data()


cars_data = get_cars_data()

INITIAL_GREETING = (
    "Hello! 👋 I'm your FindYourCar advisor. Tell me a bit about who you're buying "
    "the car for and what kind of vibe or daily purpose you have in mind "
    "(e.g., family trips with kids, a sporty head-turner, a comfortable Autobahn cruiser, or a practical city car)!"
)

# ---------------------------------------------------------
# 1. SESSION STATE INITIALIZATION
# ---------------------------------------------------------
if "baseline_confirmed" not in st.session_state:
    st.session_state.baseline_confirmed = False

if "budget_val" not in st.session_state:
    st.session_state.budget_val = 25000

if "annual_km_val" not in st.session_state:
    st.session_state.annual_km_val = 15000

if "transmission_val" not in st.session_state:
    st.session_state.transmission_val = "automatic"

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": INITIAL_GREETING}
    ]

if "extracted_slots" not in st.session_state:
    st.session_state.extracted_slots = DEFAULT_SLOTS.copy()

if "profile_ready" not in st.session_state:
    st.session_state.profile_ready = False

if "recommendations" not in st.session_state:
    st.session_state.recommendations = None


def reset_all():
    st.session_state.baseline_confirmed = False
    st.session_state.messages = [{"role": "assistant", "content": INITIAL_GREETING}]
    st.session_state.extracted_slots = DEFAULT_SLOTS.copy()
    st.session_state.profile_ready = False
    st.session_state.recommendations = None


# ---------------------------------------------------------
# 2. SIDEBAR (ACTIVE AFTER BASELINE IS CONFIRMED)
# ---------------------------------------------------------
with st.sidebar:
    if os.path.exists("logo.jpg"):
        st.image("logo.jpg", use_container_width=True)
    else:
        st.title("🚘 FindYourCar")

    if st.session_state.baseline_confirmed:
        st.caption("Baseline constraints (adjust anytime)")

        with st.container(border=True):
            st.markdown("**💰 Budget & Distance**")
            st.session_state.budget_val = st.slider(
                "Max Purchase Price",
                min_value=3000,
                max_value=80000,
                value=int(st.session_state.budget_val),
                step=500,
                format="€%d"
            )
            st.session_state.annual_km_val = st.select_slider(
                "Yearly Driving Distance",
                options=[5000, 10000, 15000, 20000, 25000, 30000, 40000],
                value=int(st.session_state.annual_km_val),
                format_func=lambda x: f"{x:,} km / yr"
            )

        with st.container(border=True):
            st.markdown("**⚙️ Transmission**")
            tx_options = ["any", "automatic", "manual"]
            st.session_state.transmission_val = st.radio(
                "Gearbox Preference",
                options=tx_options,
                index=tx_options.index(st.session_state.transmission_val),
                label_visibility="collapsed",
                format_func=lambda x: {
                    "any": "No preference (Any)",
                    "automatic": "Automatic only",
                    "manual": "Manual only"
                }[x]
            )

        st.write("")
        if st.button("🔄 Start Over", use_container_width=True):
            reset_all()
            st.rerun()
    else:
        st.info("👈 Complete **Step 1** on the main screen to unlock your advisor chat.")

# ---------------------------------------------------------
# 3. HERO BANNER
# ---------------------------------------------------------
st.markdown("""
<div class="hero-card">
    <h1>FindYourCar</h1>
    <p>Data-driven car recommendations for the German market. Share your lifestyle and vibe, and we'll calculate the best matching vehicle models with real-world ownership costs (TCO).</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 4. STEP 1: MANDATORY BASELINE ONBOARDING WIZARD
# ---------------------------------------------------------
if not st.session_state.baseline_confirmed:
    st.subheader("Step 1: Set Your Baseline")
    st.write("Before chatting with the AI Advisor, let's lock in your core financial and driving basics:")

    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### 💰 Max Budget")
            wiz_budget = st.slider(
                "Maximum Purchase Price (€)",
                min_value=3000,
                max_value=80000,
                value=int(st.session_state.budget_val),
                step=500,
                format="€%d"
            )
        with c2:
            st.markdown("#### 🛣️ Annual Mileage")
            wiz_km = st.select_slider(
                "How much do you drive per year?",
                options=[5000, 10000, 15000, 20000, 25000, 30000, 40000],
                value=int(st.session_state.annual_km_val),
                format_func=lambda x: f"{x:,} km / yr"
            )
        with c3:
            st.markdown("#### ⚙️ Gearbox Preference")
            tx_options = ["automatic", "manual", "any"]
            wiz_tx = st.radio(
                "Transmission",
                options=tx_options,
                index=0,
                format_func=lambda x: {
                    "automatic": "Automatic only",
                    "manual": "Manual only",
                    "any": "No preference (Any)"
                }[x]
            )

        st.write("")
        if st.button("🚘 Confirm & Start Advisor Chat", type="primary", use_container_width=True):
            st.session_state.budget_val = wiz_budget
            st.session_state.annual_km_val = wiz_km
            st.session_state.transmission_val = wiz_tx
            st.session_state.baseline_confirmed = True
            st.rerun()

# ---------------------------------------------------------
# 5. STEP 2: CONVERSATIONAL AI ADVISOR & RECOMMENDATIONS
# ---------------------------------------------------------
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Tell me about your lifestyle, family, or the vibe you want..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Analyzing your lifestyle & preferences..."):
            api_history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
            turn_result = run_chat_turn(
                messages=api_history,
                current_slots=st.session_state.extracted_slots
            )

            assistant_reply = turn_result.get("reply", "Understood!")
            updated_slots = turn_result.get("extracted_profile", st.session_state.extracted_slots)
            is_ready = turn_result.get("is_profile_ready", False)

            st.session_state.extracted_slots = updated_slots
            st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

        with st.chat_message("assistant"):
            st.markdown(assistant_reply)

        if is_ready:
            st.session_state.profile_ready = True
            st.rerun()

    # ---------------------------------------------------------
    # 6. RECOMMENDATION DASHBOARD
    # ---------------------------------------------------------
    if st.session_state.profile_ready:
        st.divider()
        st.subheader("🎯 Recommended Vehicle Models for Your Lifestyle")

        with st.spinner("Analyzing German market listings..."):
            buyer_profile = create_buyer_profile(
                budget_max=float(st.session_state.budget_val),
                transmission=st.session_state.transmission_val,
                annual_km=int(st.session_state.annual_km_val),
                extracted_slots=st.session_state.extracted_slots
            )

            active_data = cars_data
            brand_warning = None

            target_brands = normalize_brand_list(
                st.session_state.extracted_slots.get("preferred_brand")
            )
            if target_brands:
                mask = active_data["brand"].str.lower().isin(target_brands)
                if mask.any():
                    active_data = active_data[mask]
                else:
                    brand_warning = (
                        f"No listings found for **{', '.join(target_brands).title()}** in the dataset. "
                        "Showing the best matches across all brands instead."
                    )

            recs, info = recommend(buyer_profile, data=active_data, k=5)

            # Fallback if preferred brand had 0 matches within budget/constraints
            if recs.empty and target_brands:
                brand_warning = (
                    f"No matching **{', '.join(target_brands).title()}** models found within your "
                    f"€{st.session_state.budget_val:,} budget and criteria. Showing top alternatives across all makes:"
                )
                recs, info = recommend(buyer_profile, data=cars_data, k=5)

            st.session_state.recommendations = recs

        if brand_warning:
            st.info(brand_warning)

        if recs.empty:
            st.warning("No matching vehicles found within these exact parameters. Try raising your budget or loosening your preferences in the chat.")
        else:
            for idx, (_, r) in enumerate(recs.iterrows(), 1):
                with st.container(border=True):
                    col1, col2, col3 = st.columns([3, 2, 2])

                    with col1:
                        st.markdown(f"### {idx}. {r['brand'].title()} {r['model']}")
                        st.write(
                            f"**Typical Profile:** ~{int(r['typical_year'])}, "
                            f"{int(r['typical_km']):,} km | {r['body_type']} | "
                            f"{r['fuel_group']} ({r['transmission_group']}) | "
                            f"~{int(r['typical_power_ps'])} PS"
                        )
                        st.caption(f"Market liquidity: **{int(r['listings_count'])} active listings analyzed**")

                    with col2:
                        st.metric(
                            label="Typical Price",
                            value=f"€{r['typical_price']:,.0f}",
                            help=f"Middle 50% price range: €{r['price_q25']:,.0f} - €{r['price_q75']:,.0f}."
                        )
                        st.caption(f"IQR: €{r['price_q25']:,.0f} – €{r['price_q75']:,.0f}")

                    with col3:
                        monthly_tco = r['cost_total'] / 12
                        st.metric(
                            label="Est. Monthly TCO",
                            value=f"€{monthly_tco:,.0f} / mo",
                            help=f"Annual total: €{r['cost_total']:,.0f}/year including depreciation, fuel, tax, maintenance, and insurance."
                        )
                        st.caption(f"~€{r['cost_total']:,.0f} / year all-in")

                    reasons = r["reasons"]
                    if reasons["why"]:
                        st.markdown("**Why it fits:**")
                        for w in reasons["why"]:
                            st.markdown(f"- ✅ {w}")

                    if reasons["warn"]:
                        st.markdown("**Good to know:**")
                        for w in reasons["warn"]:
                            st.markdown(f"- ℹ️ {w}")

                    with st.expander("📊 Detailed Annual Cost Breakdown (TCO)"):
                        b1, b2, b3, b4, b5 = st.columns(5)
                        b1.metric("Fuel", f"€{r['cost_fuel']:,.0f}")
                        b2.metric("Depreciation", f"€{r['cost_depreciation']:,.0f}")
                        b3.metric("Maintenance", f"€{r['cost_maintenance']:,.0f}")
                        b4.metric("Insurance", f"€{r['cost_insurance']:,.0f}")
                        b5.metric("Vehicle Tax", f"€{r['cost_tax']:,.0f}")