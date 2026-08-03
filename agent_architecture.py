"""
CreativeSync Multi-Agent State Machine Root Entry Point
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.agents.agent_architecture import app, AgentState

if __name__ == "__main__":
    print("==================================================================")
    print(" CreativeSync FAISS RAG Multi-Agent State Machine Entry Point")
    print("==================================================================")

    test_inquiry = (
        "Client Name: Kasun Perera, Phone: +94 77 123 4567, Shoot Type: Casual, Family & Portrait Shoots, "
        "Location: Colombo, Sri Lanka, Date: 2026-08-15, Time: 18:00:00, Aesthetic: Dramatic Golden Hour & Sunset Fill Light, "
        "B2B Studio Booking: FALSE (Retail End-Client Rates), Add-ons Selected: 24-Hour Express Edit (+LKR 10,000), "
        "Notes: Outdoor portrait session with fill light and fast turnaround."
    )

    initial_state: AgentState = {
        "client_inquiry": test_inquiry,
        "parsed_intent": "",
        "location_logistics": "",
        "retrieved_context": "",
        "proposal_draft": "",
        "final_proposal": "",
        "messages": [],
    }

    try:
        final_state = app.invoke(initial_state)
        print("\n================================================ metaphysics")
        print(" WORKFLOW EXECUTION COMPLETE")
        print("================================================ metaphysics\n")
        print("--- FINAL PROPOSAL DELIVERABLE ---")
        print(final_state.get("final_proposal", "No proposal generated."))
    except Exception as e:
        print(f"[EXECUTION ERROR]: {e}")
