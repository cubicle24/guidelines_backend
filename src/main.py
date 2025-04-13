import os
from dotenv import load_dotenv
from Guidelines import Guidelines

# Load environment variables from .env file if present
load_dotenv()

def main():
    # Initialize the system
    screening_system = Guidelines()
    
    # Example clinical note
    clinical_note = """Patient is a 68 year old male. The patient smoked 35 years ago, but no longer does. He has had multiple sexual partners in the last year. 
    He has had angina in the past ten years, and required 2 stents 3 years ago. He reports being unsteady on his feet, and he walks with a cane now. 
    He reports that he has a family history of prostate cancer in his father who was 80 when diagnosed. His mother had low bone density at age 65. His blood pressure today is 180/85, but he feels fine.
    He mentions that he is unsure of his future and how fast the world is changing. This keeps him up at night, and he doesn't want to go out with friends anymore and gets nervous around crowds now due to 
    feeling unsafe."""
    
    # Generate recommendations
    results = screening_system.generate_recommendations({"clinical_note": clinical_note})
    
    # Print recommendations
    print("\nRECOMMENDED SCREENING TESTS:")
    print(results["recommendations"])
    print("\n")

if __name__ == "__main__":
    main()