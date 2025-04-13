import re
from chromadb import api
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma  
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnableSequence  # Import RunnableSequence
import os
from langchain_core.output_parsers import JsonOutputParser
from GuidelinesLoader import GuidelinesLoader
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings


class SentenceTransformerEmbeddings(Embeddings):
    """Wrapper for sentence_transformers embeddings to work with LangChain."""
    
    def __init__(self, model_name):
        """Initialize with a model name."""
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts):
        """Embed a list of documents."""
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text):
        """Embed a query."""
        return self.model.encode(text).tolist()

class Guidelines:
    def __init__(self, api_key=None):
        """Initialize the RAG system with necessary components."""
        # self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro-exp-03-25", temperature=0.0)
        self.choose_model("llama4")
        # self.embeddings = SentenceTransformerEmbeddings('all-MiniLM-L6-v2')
        self.setup_vector_db()
        self.setup_rag_pipeline()

    def choose_model(self, model_name):
        """Set up the model and embeddings based on the model name. Allows you to switch LLMs"""
        if model_name == "google":
            print(f"selecting google model")
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0)
            self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=os.environ.get("GOOGLE_API_KEY"))
        else:
            print(f"selecting llama4 model")
            groq_api_key = os.environ.get("GROQ_API_KEY")
            self.llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.0, api_key="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")
            # self.embeddings = SentenceTransformerEmbeddings('all-mpnet-base-v2')
            self.embeddings = SentenceTransformerEmbeddings('emilyalsentzer/Bio_ClinicalBERT')
            # self.embeddings = SentenceTransformerEmbeddings('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
            # self.embeddings = SentenceTransformerEmbeddings('dmis-lab/biobert-base-cased-v1.1')
    def setup_vector_db(self):
        """Load and index clinical guidelines into a vector database with LLM-extracted metadata."""
        try:
            # raise Exception("Forcing an error to test the except block")
            # Try to load existing vector database
            self.vector_db = Chroma(
                collection_name="preventive_guidelines",
                embedding_function=self.embeddings,  # Use our wrapper class
                persist_directory="../guidelines_db",
            )
            print("Loaded existing guideline database")
        except:
            print("Creating new guideline database with LLM metadata extraction")
            
            # Use the LLM-based metadata extractor
            loader = GuidelinesLoader("../guidelines_repository", llm=self.llm)
            documents = loader.load()
            
            print(f"Loaded {len(documents)} guideline documents with LLM-extracted metadata")
            
            # Split documents while preserving metadata
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            splits = text_splitter.split_documents(documents)
            
            print(f"Split into {len(splits)} chunks for indexing")
            
            # Create vector database
            self.vector_db = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,  # Use our wrapper class
                collection_name="preventive_guidelines",
                persist_directory="../guidelines_db"
            )
    
    def create_extraction_prompt(self):
        """Set up LLM chain for extracting patient information from a clinical note."""
        extraction_prompt = PromptTemplate(
            input_variables=["clinical_note"],
            template="""
            Your task is to extract a patient's medical information into a structured JSON object from a clinical note:

            Clinical Note:
            {clinical_note}

            Extract the following information and format the output as valid JSON. If a value for field is not available, make the value "null":

            1. Patient age (as a number)
            2. Patient gender (male/female)
            3. Race
            4. Past medical history (list of conditions)
            5. Family history (list of conditions and affected relatives)
            6. Social history (including smoking status, alcohol use, drug use, exercise, occupation, travel history)
            7. Previous screening tests with dates (e.g., colonoscopy, mammogram)

            """
        )
            # Example of the desired structured JSON output:
            # {'patient_age': 52, 
            # 'patient_gender': 'female', 
            # 'race': None, 
            # 'past_medical_history': ['hypertension', 'type 2 diabetes', 'mild depression'], 
            # 'family_history': ['breast cancer in mother (diagnosed at age 49)', 'colorectal cancer in father (age 62)'], 
            # 'social_history': {'smoking_status': 'former smoker, quit 5 years ago', 'alcohol_use': 'drinks alcohol socially (2-3 drinks/week)', 'drug_use': None, 'exercise': 'moderate exercise 2x weekly', 'occupation': None, 'recent_travel_history': None, 'pack_year_history': '30'}, 
            # 'previous_screening_tests': {'colonoscopy': '2018 (normal)', 'mammogram': '2021 (BIRADS 2)', 'pap_smear': '2020 (normal)', 'lipid_panel': '1 year ago (borderline high LDL)'}}
        return extraction_prompt

    def create_recommendation_prompt(self):
        """Set up chain for generating screening recommendations."""
        recommendation_prompt = PromptTemplate(
            input_variables=["patient_data", "guidelines"],
            template="""
            Based on the patient information and relevant clinical guidelines for screening tests, recommend appropriate screening tests for the patient.
            Format as a ranked list with justification and next due date for each test.  The recommendations should be ordered from highest priority to lowest priority.

            You must recommend all screening tests that the patient meets the criteria for based on the patient's age or gender. Do not miss them. 
            Never recommend a screening test for a male that only applies to females, and vice versa.

            Do not make up information.  Do not query the internet.  Only give recommendations based on the retieved documents 
            and what is explicitly state or can be reasonably inferred.  If a test is not indicated, do not mention it at all.

            If you can not make any recommendations based on the guidelines and the patient's clinical information, state that you can not make any recommendations.
            You may only use the clinical guidelines and not your general knowledge to make recommendations.

            Patient Information:
            {patient_data}
            
            Relevant Guidelines:
            {guidelines}
            
            List all recommendations that you find to be appropriate.  Limit the justification and evidence for each recommendation to 20 words or less
            For the evidence, quote the most relevant parts of the guidelines that you used to make the recommendation.
            
            Provide recommendations in this format:
            1. [Test Name] - [Justification] - Next due: [Date] - Evidence: [Evidence]
            2. [Test Name] - [Justification] - Next due: [Date] - Evidence: [Evidence]
            3. [Test Name] - [Justification] - Next due: [Date] - Evidence: [Evidence]

            After first looking only at the provided guidelines and giving recommendations, add another set of recommendations.  For this section, read through the full clinical note provided
            and ONLY search the US Preventive Task Force Guidelines website for additional screening tests that the patient meets criteria for. You are not allowed to search any other website.
            Make sure to say that these are only suggestions that were found online and are not guaranteed to be relevant.  The clinician must review these additional recommendations with more scrutiny.
            Make as many additional recommendations as you can find reasonable justification for.  Limit the justification and evidence for each recommendation to 20 words or less.

            If a test is not indicated, do not mention it at all.

            Provide additional recommendations in this format:
            -----------------------------------------
            ---POSSIBLE ADDITIONAL RECOMMENDATIONS---
            -----------------------------------------

            1. [Test Name] - [Justification] - Next due: [Date] - Evidence: [Evidence]
            2. [Test Name] - [Justification] - Next due: [Date] - Evidence: [Evidence]
            3. [Test Name] - [Justification] - Next due: [Date] - Evidence: [Evidence]

            """
        )
        return recommendation_prompt



    def create_query_from_patient_data(self, patient_data):
        """Create a custom query to retrieve the most relevant guidelines based on each patient's medical history."""
    
        query_parts = []
        
        # Add demographics
        demographics = f"{patient_data.get('patient_age', '')} year old {patient_data.get('patient_gender', '')}"
        query_parts.append(demographics)
        
        # Add past medical history
        if patient_data.get('past_medical_history'):
            conditions = ", ".join(patient_data['past_medical_history'])  # Limit to most relevant
            query_parts.append(f"a past medical history of {conditions}")
        
        # Add family history
        # if patient_data.get('family_history'):
        #     family = ", ".join(patient_data['family_history'])
        #     query_parts.append(f"family history of {family}")
        
        # Add social history
        if patient_data.get('social_history'):
            social = ", ".join(patient_data['social_history'])
            query_parts.append(f"{social}")
        
        # Join all parts with natural language connectors
        query = " with ".join(query_parts) + ". Look for all applicable preventive screening guidelines."
        
        print(f"Query: {query}")
        return query
    
    def setup_rag_pipeline(self):
        """Set up the integrated RAG pipeline using LCEL."""
        
        # Create a retriever from the vector database
        base_retriever = self.vector_db.as_retriever(
            search_kwargs={"k": 75})
        compressor = LLMChainExtractor.from_llm(self.llm)
        retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)
    
        extraction_prompt = self.create_extraction_prompt()
        recommendation_prompt = self.create_recommendation_prompt()

        def debug_patient_data(patient_data):
            """Helper method to debug patient data"""
            print("\n==== DEBUG: PATIENT DATA ====")
            print(f"Type: {type(patient_data)}")
            print(f"Content: {patient_data}")
            print("============================\n")
            return patient_data

        debug_component = RunnableLambda(debug_patient_data)


        # Define retrieval step as a named function or component
        def retrieve_and_format_guidelines(inputs):
            # query = self.create_query_from_patient_data(inputs["patient_data"])
            # query = "52 year old female with past medical history of: black eyes, bruises on her arms, feels sad, low appetite. Social history of smoking_status: smokes, alcohol_use: drinks 5 beers per day. Previous_screening_tests of: low-dose CT scan of chest"
            query = """Patient is a 52 year old female. The patient smokes, drinks 5 beers per day.  She currently has been coughing for at least 2 months.
    She had a low-dose CT scan of her chest six months ago that was normal. Her last colonoscopy was 1 year ago, and it was normal.
    She reports feeling sad and without purpose after her husband passed away six months ago.  She is not eating well and is sleeping very little.
    The patient's mother died of breast cancer at age 45. The patient has a black eye, and bruises on her arms. Her blood pressure is 190/100 today. She is overweight now, having gained 15 pounds since
    last visit.  She reports eating lots of carbohydrates and junk food."""

            # query = """Patient is a 68 year old male. The patient smoked 35 years ago, but no longer does.  He has had multiple sexual partners in the last year. 
            # He has had angina in the past ten years, and required 2 stents 3 years ago.  He reports being unsteady on his feet, and he walks with a cane now. 
            # He reports that he has a family history of prostate cancer in his father who was 80 when diagnosed.  His mother had low bone density at age 65.  His blood pressure today is 180/85, but he feels fine.
            # He mentions that he is unsure of his future and how fast the world is changing. This keeps him up at night, and he doesn't want to go out with friends anymore and gets nervous around crowds now due to 
            # feeling unsafe."""
            relevant_docs = retriever.invoke(query)
            return {
                "patient_data": inputs["patient_data"],
                "guidelines": "\n\n".join([doc.page_content for doc in relevant_docs])
            }
        # Create a named component
        guidelines_fetcher = RunnableLambda(retrieve_and_format_guidelines)
        
        # Build the LCEL pipeline with named components
        self.rag_pipeline = (
            # Start with clinical note
            # {"clinical_note": lambda x: x}
            # Extract patient information using JSON output parser
            {"patient_data": extraction_prompt | self.llm | JsonOutputParser() }

            | debug_component
            # Retrieve relevant guidelines
            | guidelines_fetcher
            # Generate recommendations
            | {"patient_data": lambda x: x["patient_data"],
               "recommendations": recommendation_prompt | self.llm}
        )


    def generate_recommendations(self, clinical_note):
        """End-to-end process to generate screening recommendations using LCEL pipeline."""
        # Process the clinical note through the RAG pipeline
        result = self.rag_pipeline.invoke(clinical_note)
        # print(f"final result: {result}")
        # Print intermediate results for debugging
        # print(f"Extracted patient data: {result['patient_data']}")
        
        return {
            "patient_data": result["patient_data"],
            "recommendations": result["recommendations"].content
        }

# Example usage
def main():
    # Initialize the system - no API key needed as we're using GOOGLE_API_KEY from environment
    screening_system = Guidelines()
    
    # Example clinical note
    # clinical_note = """
    # Patient is a 52-year-old female presenting for annual check-up. 
    # Past medical history significant for hypertension, type 2 diabetes (diagnosed 2019), and mild depression.
    # Family history notable for breast cancer in mother (diagnosed at age 49) and colorectal cancer in father (age 62).
    # Social history: Former smoker, quit 5 years ago. 30 pack-year history. Drinks alcohol socially (2-3 drinks/week).
    # Gets moderate exercise 2x weekly.
    # Last colonoscopy was in 2018 (normal). Last mammogram in 2021 (BIRADS 2). 
    # Last Pap smear in 2020 (normal). Last lipid panel 1 year ago (borderline high LDL).
    # """
    clinical_note =   """Patient is a 52 year old female. The patient smokes, drinks 5 beers per day.  She currently has been coughing for at least 2 months.
    She had a low-dose CT scan of her chest six months ago that was normal. Her last colonoscopy was 1 year ago, and it was normal.
    She reports feeling sad and without purpose after her husband passed away six months ago.  She is not eating well and is sleeping very little.
    The patient's mother died of breast cancer at age 45. The patient has a black eye, and bruises on her arms. Her blood pressure is 190/100 today. She is overweight now, having gained 15 pounds since
    last visit.  She reports eating lots of carbohydrates and junk food."""
    
    # clinical_note =   """Patient is a 68 year old male. The patient smoked 35 years ago, but no longer does.  He has had multiple sexual partners in the last year. 
    # He has had angina in the past ten years, and required 2 stents 3 years ago.  He reports being unsteady on his feet, and he walks with a cane now. 
    # He reports that he has a family history of prostate cancer in his father who was 80 when diagnosed.  His mother had low bone density at age 65.  His blood pressure today is 180/85, but he feels fine.
    # He mentions that he is unsure of his future and how fast the world is changing. This keeps him up at night, and he doesn't want to go out with friends anymore and gets nervous around crowds now due to 
    # feeling unsafe."""
    # Generate recommendations
    results = screening_system.generate_recommendations({"clinical_note" : clinical_note})
    
    # Print recommendations
    print("\nRECOMMENDED SCREENING TESTS:")
    print(results["recommendations"])
    print("\n")



if __name__ == "__main__":
    main()