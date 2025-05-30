import json
from typing import Mapping
from sdoh_agent import run_agent, AgentState

def main():
    """Demonstrates the SDOH agent"""

    #load note:
    note = open("../clinical_notes_repository/note_1_medicine_CHF.txt").read()
    # note = open("../clinical_notes_repository/easy_note.txt").read()
    start_state: AgentState = {
        "note": note,
        "sdoh": {},
        "intervention": {}
    }
    end_state = run_agent(note, start_state)
    print(f"Final state: {end_state}")

if __name__ == "__main__":
    main()