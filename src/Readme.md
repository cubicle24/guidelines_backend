# Clinical Guidelines RAG System

A Retrieval-Augmented Generation (RAG) system for clinical screening guidelines.

Motivation: Screening and prevention of disease is still the most effective treatment we have.  It also saves the most money.
However, there are at least 97 screening guidelines (which frequently change) that healthcare providers are expected to remember.
This is not realistic for providers, and it results in many patients not receiving the care that the evidence suggests.
This clinical decision support system intelligently parses a patient's clinical note(s) and retrieves relevant screening guidelines.

This system is for educational purposes and does not replace professional medical advice.

## Features

- Extracts patient information from clinical notes
- Retrieves relevant clinical guidelines based on patient data
- Generates personalized screening recommendations
- Supports multiple LLM providers (Google Gemini, Groq/Llama)

## Setup

1. Install dependencies: pip install -r requirements.txt
2. Make a virtual environment: python3 -m venv guidelines_rag (you can use any name you want;
   here it is "guidelines_rag")
3. Activate the virtual environment you just created: source guidelines_rag/bin/activate
2. Set up environment variables:
    - `OPENAI_API_KEY`: Your OpenAI API key
    - `GEMINI_API_KEY`: Your Google Gemini API key
    - `GROQ_API_KEY`: Your Groq API key
3. Switch into the `src` directory: cd src
4. Run the main script: python main.py

## Usage

1. Provide a patient's clinical notes when prompted.
2. The system will extract patient information and retrieve relevant screening test guidelines 
based on the US Preventive Task Force recommendations.
3. The generated recommendations will be displayed. Screening tests that are based only on the 
local guidelines repository will be given first; additional ones based on a search of medical literature will also be displayed.

## API Usage

The system provides a REST API for integration with other applications:

1. Start the API server: `python api.py`
2. The API will be available at http://localhost:8000
3. API endpoints:
   - POST `/api/recommendations`: You give the system a clinical note, and screening recommendations will be returned
   - GET `/api/hello`: makes sure the API is running

API documentation is available at http://localhost:8000/docs when the server is running.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
This project is licensed under the MIT License.
