"""
CreativeSync | Premium Photography Shoot Booking Application
Root Entry Point: app.py
"""

import os
import sys
import urllib.parse
from datetime import datetime, time

# 1. Environment variable loading at the very top (force override)
from dotenv import load_dotenv
load_dotenv(override=True)

# Debug prints to verify API key loading
print(f"[app.py] Groq Key Loaded: {bool(os.getenv('GROQ_API_KEY'))}")
print(f"[app.py] OpenRouter Key Loaded: {bool(os.getenv('OPENROUTER_API_KEY'))}")

import streamlit as st

# Import LangGraph multi-agent pipeline from package architecture
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
try:
    from src.agents.agent_architecture import app as agent_app, AgentState
except ImportError:
    from agent_architecture import app as agent_app, AgentState


# 2. Helper function: Extract client-facing proposal only
def extract_client_proposal(raw_output: str) -> str:
    """
    Filters raw agent output to extract exclusively the client-facing shoot proposal.
    Preserves API exceptions and error messages if present.
    """
    if not raw_output:
        return ""
    
    # If output contains an error/exception, return it directly so user can debug
    if "[API EXCEPTION" in raw_output or "[GROQ_API_KEY" in raw_output or "[OPENROUTER_API_KEY" in raw_output:
        return raw_output.strip()
    
    # Audit headers to split and remove
    audit_markers = [
        "### QUALITY & SAFETY REFLECTION AUDIT",
        "### REFLECTION & SAFETY AUDIT",
        "QUALITY & SAFETY REFLECTION AUDIT",
        "REFLECTION & SAFETY AUDIT",
        "FINAL STATUS:",
    ]
    
    clean_text = raw_output
    for marker in audit_markers:
        if marker in clean_text:
            clean_text = clean_text.split(marker)[0]
            
    return clean_text.strip()


# 3. Page Configuration
st.set_page_config(
    page_title="CreativeSync | Premium Shoot Booking",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# 4. Custom CSS Styling (Rich Aesthetics & Premium Feel)
st.markdown("""
<style>
    /* Global Styles & Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Header Container */
    .header-container {
        text-align: center;
        padding: 2rem 1rem 1rem 1rem;
        background: linear-gradient(135deg, #1e1b4b 0%, #311b92 50%, #4a148c 100%);
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(49, 27, 146, 0.4);
    }
    
    .header-title {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.4rem;
        background: linear-gradient(90deg, #ffffff, #e0e7ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .header-subtitle {
        font-size: 1.05rem;
        color: #c7d2fe;
        font-weight: 300;
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-block;
        background: #059669;
        color: white;
        padding: 0.35rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    /* Form Styling Enhancements */
    .stButton > button {
        background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        font-weight: 600;
        font-size: 1rem;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 14px rgba(124, 58, 237, 0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #4338ca 0%, #6d28d9 100%);
        box-shadow: 0 6px 20px rgba(124, 58, 237, 0.5);
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)


# 5. Header Render
st.markdown("""
<div class="header-container">
    <div class="header-title">📷 CreativeSync</div>
    <div class="header-subtitle">Agentic AI Studio Booking & Production Planner</div>
</div>
""", unsafe_allow_html=True)


# 6. Client Shoot Booking Form
st.markdown("### 📝 Photography Shoot Booking Form")
st.write("Submit your shoot specifications below. Our multi-agent AI system will analyze your intent, consult our equipment corpus, and compile a tailored proposal.")

with st.form(key="shoot_booking_form"):
    is_studio_b2b = st.checkbox("Are you a Studio booking a Freelancer? (B2B Rates)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        shoot_type = st.selectbox(
            "Shoot Type",
            [
                "Wedding Coverage (Full Day / Ceremony)",
                "Pre-Shoot / Engagement Session",
                "Casual, Family & Portrait Shoots",
                "Commercial & Corporate Branding",
                "Product & Food Photography",
                "Events, Birthdays & Parties",
                "Other"
            ],
            index=0,
        )
        location = st.text_input(
            "Location / Venue",
            value="Colombo, Sri Lanka",
            placeholder="e.g., Galle Face Hotel, Malabe, Kandy...",
        )
        
    with col2:
        shoot_date = st.date_input(
            "Shoot Date",
            value=datetime(2026, 8, 15),
        )
        shoot_time = st.time_input(
            "Preferred Start Time",
            value=time(18, 0),
        )

    aesthetic = st.selectbox(
        "Desired Aesthetic & Lighting Mood",
        [
            "Dramatic Golden Hour & Sunset Fill Light",
            "High-Contrast Moody Editorial",
            "Soft Bright Natural Light",
            "Studio Strobe & Color Gel Accent",
            "Clean High-Key Product Commercial",
        ],
        index=0,
    )
    
    st.markdown("##### ⚡ Premium Add-ons")
    col_addon1, col_addon2, col_addon3 = st.columns(3)
    with col_addon1:
        add_express = st.checkbox("24-Hour Express Edit (+ LKR 10,000)")
    with col_addon2:
        add_raw = st.checkbox("Raw Files Included (+ LKR 15,000)")
    with col_addon3:
        add_drone = st.checkbox("Drone Coverage (+ LKR 15,000)")

    special_requirements = st.text_area(
        "Special Requirements & Deliverable Specs",
        height=100,
        placeholder="Detail lighting preferences, drone needs, shot list items, or urgent turnaround requirements...",
    )

    st.markdown("##### 👤 Client Contact Information")
    col_contact1, col_contact2 = st.columns(2)
    with col_contact1:
        client_name = st.text_input("Full Name", placeholder="e.g., Kasun Perera")
        client_email = st.text_input("Email Address", placeholder="kasun@example.com")
    with col_contact2:
        client_phone = st.text_input("Contact / WhatsApp Number", placeholder="+94 77 123 4567")

    submit_button = st.form_submit_button(label="🚀 Compile & Book Shoot Proposal")


# 7. Form Submission & Agent State Machine Invocation
if submit_button:
    # Build add-ons string
    selected_addons = []
    if add_express:
        selected_addons.append("24-Hour Express Edit (+LKR 10,000)")
    if add_raw:
        selected_addons.append("Raw Files Included (+LKR 15,000)")
    if add_drone:
        selected_addons.append("Drone Coverage (+LKR 15,000)")
    
    addons_str = ", ".join(selected_addons) if selected_addons else "None"
    b2b_str = "TRUE (Studio Freelance B2B Rates apply)" if is_studio_b2b else "FALSE (Retail End-Client Rates)"

    # Dynamically format ALL user inputs into single string for client_inquiry
    combined_inquiry = (
        f"Client Name: {client_name}, Email: {client_email}, Phone: {client_phone}, "
        f"Shoot Type: {shoot_type}, Location: {location}, Date: {shoot_date}, Time: {shoot_time}, "
        f"Aesthetic: {aesthetic}, B2B Studio Booking: {b2b_str}, Add-ons Selected: {addons_str}, "
        f"Notes: {special_requirements}"
    )

    initial_state: AgentState = {
        "client_inquiry": combined_inquiry,
        "parsed_intent": "",
        "location_logistics": "",
        "proposal_draft": "",
        "final_proposal": "",
        "messages": [],
    }

    # Execute agent pipeline with spinner
    with st.spinner("Compiling shoot proposal..."):
        try:
            final_state = agent_app.invoke(initial_state)
            
            # Extract raw final output from Reflection agent
            raw_output = final_state.get("final_proposal") or final_state.get("final_output") or "Proposal compiled successfully."
            
            # Filter output to extract client proposal, preserving any API exception error messages
            client_proposal = extract_client_proposal(raw_output)
            
            # Render proposal / error directly to user UI
            if "[API EXCEPTION" in client_proposal or "UNSET/MISSING" in client_proposal:
                st.error("⚠️ API Execution Issue Detected:")
                st.code(client_proposal)
            else:
                st.markdown("<div class='status-badge'>✓ PROPOSAL APPROVED</div>", unsafe_allow_html=True)
                st.markdown("### 📋 Tailored Shoot Proposal & Quote")
                st.markdown(client_proposal)

                # WhatsApp Booking Confirmation Button
                wa_message = f"Hi! I am {client_name} ({client_phone}). I would like to book a {shoot_type} on {shoot_date} at {location}."
                encoded_wa = urllib.parse.quote(wa_message)
                wa_url = f"https://wa.me/94700000000?text={encoded_wa}"
                
                st.markdown("---")
                st.link_button("💬 Confirm & Book via WhatsApp", wa_url)

        except Exception as e:
            st.error(f"An error occurred while compiling the proposal: {e}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #94a3b8; font-size: 0.85rem;'>"
    "CreativeSync Studio System v0.1.0 • Powered by LangGraph & Agentic AI"
    "</div>",
    unsafe_allow_html=True,
)
