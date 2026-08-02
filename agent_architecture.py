"""
Root entry point for CreativeSync 3-Node Multi-Agent State Machine.
Imports and executes from src/agents/agent_architecture.py.
"""

from src.agents.agent_architecture import (
    AgentState,
    StudioState,
    router_node,
    orchestrator_node,
    reflection_node,
    build_agent_graph,
    app,
)

if __name__ == "__main__":
    print("==================================================================")
    print(" CreativeSync 3-Node Multi-Agent State Machine (Root Entry Point)")
    print("==================================================================")

    sample_inquiry = (
        "Hi CreativeSync! I'm looking to book a 2-hour outdoor sunset portrait shoot at "
        "Malibu Beach on August 15th around 6:00 PM. I need dramatic golden hour lighting with fill light, "
        "professional gear, and fast turnaround on retouched photos."
    )

    initial_state: AgentState = {
        "client_inquiry": sample_inquiry,
        "parsed_intent": "",
        "proposal_draft": "",
        "final_proposal": "",
        "messages": [],
    }

    final_state = app.invoke(initial_state)

    print("\n==================================================================")
    print(" WORKFLOW EXECUTION COMPLETE")
    print("==================================================================")
    print("\n--- STRUCTURED MESSAGE EXCHANGE ---")
    for idx, msg in enumerate(final_state.get("messages", []), 1):
        print(f"\n[{idx}] {msg['sender']}:")
        print(msg['content'])

    print("\n==================================================================")
    print("--- FINAL PROPOSAL DELIVERABLE ---")
    print(final_state.get("final_proposal", "No final proposal generated."))
