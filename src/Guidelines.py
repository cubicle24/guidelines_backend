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
from .GuidelinesLoader import GuidelinesLoader
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from langchain_fireworks import ChatFireworks
import shutil
from dotenv import load_dotenv


load_dotenv()

#without these abs paths, won't deploy correctly  on cloud servers that don't resolve relative paths well
script_dir = os.path.dirname(os.path.realpath(__file__))
guidelines_repo_path = os.path.abspath(os.path.join(script_dir, "../guidelines_repository"))
persist_db_path = os.path.abspath(os.path.join(script_dir, "../guidelines_db"))
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
        self.choose_model("google")
        self.setup_vector_db()
        self.setup_rag_pipeline()

    def choose_model(self, model_name):
        """Set up the model and embeddings based on the model name. Allows you to switch LLMs"""
        if model_name == "google":
            print(f"selecting google model")
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not google_api_key:
                raise ValueError("GOOGLE_API_KEY environment variable is not found.")
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", temperature=0.0, google_api_key=google_api_key)
            # self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro-exp-03-25", temperature=0.0)
            # self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=os.environ.get("GOOGLE_API_KEY"))
            # self.embeddings = SentenceTransformerEmbeddings('emilyalsentzer/Bio_ClinicalBERT')
            # model_small = SentenceTransformer("abhinand/MedEmbed-small-v0.1")
            self.embeddings = SentenceTransformerEmbeddings("abhinand/MedEmbed-base-v0.1")
            # model_large = SentenceTransformer("abhinand/MedEmbed-large-v0.1")
        elif model_name == "deepseek":
            print(f"selecting deepseek model")
            self.llm = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0.0)
            self.embeddings = SentenceTransformerEmbeddings('emilyalsentzer/Bio_ClinicalBERT')
        elif model_name == "llama3":
            print(f"selecting llama3 model")
            self.llm = ChatGroq(model="llama3-70b-8192", temperature=0.0)
            self.embeddings = SentenceTransformerEmbeddings('emilyalsentzer/Bio_ClinicalBERT')
        elif model_name == "llama3_fireworks":
            print(f"selecting llama3 fireworks model")
            self.llm = ChatFireworks(model="accounts/fireworks/models/llama-v3p3-70b-instruct", temperature=0.0)
            self.embeddings = SentenceTransformerEmbeddings('emilyalsentzer/Bio_ClinicalBERT')
        elif model_name == "llama4_fireworks":
            print(f"selecting llama4 fireworks model")
            self.llm = ChatGroq(model="accounts/fireworks/models/llama4-scout-instruct-basic", temperature=0.0)
            self.embeddings = SentenceTransformerEmbeddings('emilyalsentzer/Bio_ClinicalBERT')
        else:
            print(f"selecting lllama4 groq model")
            groq_api_key = os.environ.get("GROQ_API_KEY")
            self.llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.0)
            # self.embeddings = SentenceTransformerEmbeddings('all-mpnet-base-v2')
            self.embeddings = SentenceTransformerEmbeddings('emilyalsentzer/Bio_ClinicalBERT')
            # self.embeddings = SentenceTransformerEmbeddings('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
            # self.embeddings = SentenceTransformerEmbeddings('dmis-lab/biobert-base-cased-v1.1')
    def setup_vector_db(self):
        """Load and index clinical guidelines into a vector database with LLM-extracted metadata."""
        collection_name = "preventive_guidelines"

        # Use the absolute path variables defined above
        persist_directory = persist_db_path
        repo_directory = guidelines_repo_path

        # Check if the database directory exists and try loading
        if os.path.exists(persist_directory):
            try:
                print(f"Attempting to load existing guideline database from: {persist_directory}")
                self.vector_db = Chroma(
                    collection_name=collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=persist_directory,
                )
                # Verify collection has data
                if self.vector_db._collection.count() > 0:
                    print(f"Successfully loaded existing guideline database with {self.vector_db._collection.count()} items.")
                    return # Successfully loaded, exit the function
                else:
                    print("Existing database found but is empty. Will recreate.")
                    # Optionally remove the empty DB directory if needed
                    # shutil.rmtree(persist_directory)
            except Exception as e:
                print(f"Error loading existing database: {e}. Will recreate.")
                # Optionally remove potentially corrupted DB directory
                if os.path.exists(persist_directory):
                     try:
                         shutil.rmtree(persist_directory)
                         print(f"Removing potentially corrupted database directory: {persist_directory}")
                     except OSError as rm_error:
                         print(f"Error removing directory {persist_directory}: {rm_error}")

        # If loading failed or directory didn't exist, create a new one
        print(f"Creating new guidelines database in: {persist_directory}")
        print(f"Loading documents from: {repo_directory}") # Log the path being used

        #if you reach here, you're creating the vector db (again)
        try:
            # Use the absolute path for the loader
            loader = GuidelinesLoader(repo_directory, llm=self.llm)
            documents = loader.load()

            if not documents:
                print(f"Error: No documents were loaded from '{repo_directory}'. Please check the directory exists and contains files in the Railway deployment.")
                raise ValueError(f"Failed to load any documents from {repo_directory} for vector DB creation.")

            print(f"Loaded {len(documents)} guideline documents and extracted metadata")

            # Split documents while preserving metadata
            text_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ".", " "],
                chunk_size=1000,
                chunk_overlap=100
            )
            splits = text_splitter.split_documents(documents)

            if not splits:
                print("Error: Document splitting resulted in zero chunks. Check document content and splitter settings.")
                raise ValueError("Document splitting resulted in zero chunks.")

            print(f"Split into {len(splits)} chunks for indexing")

            # Create vector database using the absolute path
            self.vector_db = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings, # Corrected parameter name
                collection_name=collection_name,
                persist_directory=persist_directory
            )
            print("Successfully created and persisted new guidelines database.")

        except Exception as e:
            print(f"An error occurred during new database creation: {e}")
            raise

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
            3. Pregnancy status
            4. Race
            5. Past medical history (list of conditions)
            6. Family history (list of conditions and affected relatives)
            7. Social history (including smoking status, alcohol use, drug use, exercise, occupation, travel history, domestic partner violence)
            8. Previous screening tests with dates (e.g., colonoscopy, mammogram)

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
           
            <SYSTEM>:
            You are a backend API response generator. Output must be valid JSON only. Do not include any preamble, commentary, or prose. Only return a single JSON object that matches the schema provided.

            Your task is to generate evidence-based screening recommendations for a patient based on their medical information and the provided clinical guidelines.
            First, extract the patient's age, gender, and pregnancy status. These 3 pieces of information are the most important for determining which screening tests to recommend.

            Recommend all screening tests that the patient meets the criteria for, based primarily on the patient's age, gender, and pregnancy status. 
            Never recommend a screening test for a male that only applies to females, and vice versa.  If a female is not pregnant, do not recommend any pregnancy specific tests.

            Do not make up information nor query the internet nor use your general knowledge. Only consult the retrieved documents. 
            If a test is not indicated, do not mention or recommend it at all. Double check your recommendations to never make the error of recommending a test that the patient does not meet criteria for.

            If you can not make any recommendations, state that you can not make any recommendations.
            
            For each recommendation, provide your justification and reasoning for making the recommendation based on the guidelines.  Limit the justification to 100 words or less.            
            If your justification is that the patient does not fit the guidelines and are therefore not recommending a test, remove the recommendation from the list.  Do not list it all.

            Then cite the evidence (source) by finding the metadata associated with the retrieved documents. Cite the "governing_body", "topic", and "pub_date" that you consulted to make the recommendation.
            
            After first looking only at the provided guidelines and giving recommendations, attach a set of additional recommendations.
            For this section, read the entire clinical note. Then you may use your general knowledge, as well as ONLY search the US Preventive Task 
            Force Guidelines website for additional screening tests that the patient meets criteria for, focusing on age, gender, and pregnancy status. 
            Make as many additional recommendations as you can find reasonable justification for. Attach these
            additional recommendations under the JSON key "additional_recommendations" so that it is clear these come from external sources.

            Patient Information:
            {patient_data}
            
            Relevant Guidelines:
            {guidelines}

            If you have no guidelines to consult, return an empty but valid JSON object. 
            Return only the JSON object, with no extra text or formatting. Respond ONLY with valid JSON matching this schema. Do not add any extra keys or text.
            Generate recommendations as a ranked list (from highest to lowest priority).  
            Return all recommendations in this EXACT JSON structure and format below. 
            Do not include any text outside the JSON object. Double check five times that the output JSON matches the schema exactly. If not, modify the output.
           

            <EXAMPLE OUTPUT>:

            "recommendations": {{
                "recommendations": [
                    {{
                        "test": "Osteoporosis screening",
                        "justification": "Male patient with family history of low bone density",
                        "next_due_date": "Not specified",
                        "evidence": "USPSTF recommends screening for osteoporosis in women 65 years or older (moderate certainty)",
                        "governing_body": "USPSTF",
                        "topic": "Osteoporosis",
                        "pub_date": "2020"
                    }},
                    {{
                        "test": "Diabetes screening",
                        "justification": "Male patient with risk factors for diabetes (age, family history not specified)",
                        "next_due_date": "Not specified",
                        "evidence": "USPSTF recommends screening for diabetes in adults with risk factors (not specified)",
                        "governing_body": "USPSTF",
                        "topic": "Diabetes",
                        "pub_date": "Not specified"
                    }}
                ],
                        "additional_recommendations": [
                    {{
                        "test": "Osteoporosis screening",
                        "justification": "Male patient with family history of low bone density",
                        "next_due_date": "Not specified",
                        "evidence": "USPSTF recommends screening for osteoporosis in women 65 years or older (moderate certainty)",
                        "governing_body": "USPSTF",
                        "topic": "Osteoporosis",
                        "pub_date": "2020"
                    }},
                    {{
                        "test": "Hepatitis C screening",
                        "justification": "Male patient with previous IV drug use",
                        "next_due_date": "Not specified",
                        "evidence": "USPSTF recommends screening for former IV drug users",
                        "governing_body": "USPSTF",
                        "topic": "Hepatitis C Guidelines",
                        "pub_date": "2014"
                    }}
                ]

            }}

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
        
        # Creating a retriever from a vector database
        base_retriever = self.vector_db.as_retriever(
            # search_kwargs={"k":10},search_type="mmr",lambda_mult=0.1)
            search_kwargs={"k":10},search_type="similarity")
        compressor = LLMChainExtractor.from_llm(self.llm)
        # retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)
        retriever = base_retriever
    
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

        def debug_recommendation_data(rec_data):
            """Helper method to debug rec data"""
            print("\n==== DEBUG: REC DATA ====")
            print(f"Type: {type(rec_data)}")
            print(f"Content: {rec_data}")
            print("============================\n")
            return rec_data

        debug_rec_component = RunnableLambda(debug_recommendation_data)

        def retrieve_and_format_guidelines(inputs):
            print(f'inputs: {inputs}')
            query = inputs.get("clinical_note", "") + "PATIENT INFORMATION (STRUCTURED): " + str(inputs['patient_data'])
            print("\n==== RETRIEVAL DEBUG ====")
            print(f"Query sent to retriever:\n{query}\n")
            relevant_docs = retriever.invoke(query)
            print(f"Number of retrieved docs: {len(relevant_docs)}")
            # Deduplicate by page_content
            seen = set()
            unique_docs = []
            for doc in relevant_docs:
                content = doc.page_content.strip()
                if content not in seen:
                    seen.add(content)
                    unique_docs.append(doc)
            print(f"Number of unique docs after deduplication: {len(unique_docs)}")
            for i, doc in enumerate(unique_docs):
                print(f"\n--- Retrieved Doc #{i+1} ---")
                print(doc.page_content)
            print("=========================\n")
            return {
                "patient_data": inputs["patient_data"],
                "guidelines": "\n\n".join([doc.page_content for doc in unique_docs])
            }
        # Create a named component
        guidelines_fetcher = RunnableLambda(retrieve_and_format_guidelines)
        
        # Build the LCEL pipeline with named components
        self.rag_pipeline = (
            # Start with clinical note
            {"patient_data": extraction_prompt | self.llm | JsonOutputParser(),
            "clinical_note": lambda x: x["clinical_note"] }

            | debug_component
            # Retrieve relevant guidelines
            | guidelines_fetcher
            # Generate recommendations
            | {"patient_data": lambda x: x["patient_data"],
               "screening_recommendations": recommendation_prompt | self.llm | JsonOutputParser() }
            #    "screening_recommendations": recommendation_prompt | self.llm }
            # | debug_rec_component
        )


    def generate_recommendations(self, clinical_note):
        """End-to-end process to generate screening recommendations using LCEL pipeline."""
        # Process the clinical note through the RAG pipeline
        result = self.rag_pipeline.invoke(clinical_note)
        print(f"####FINAL RESULT####: {result}")

        result = self.normalize_response(result)

        return {
            "patient_data": result["patient_data"],
            "recommendations": result["screening_recommendations"]
        }

    def normalize_response(self, response_data):
        """the LLM randomly inserts an extra "recommendations" key in the output, so we need to remove it if it exists"""
        recs = response_data.get("screening_recommendations", {}).get("recommendations", {})
        if isinstance(recs, dict) and "recommendations" in recs:
            response_data["screening_recommendations"]["recommendations"] = recs["recommendations"]
            response_data["screening_recommendations"]["additional_recommendations"] = recs["additional_recommendations"]
        return response_data

def main():
    screening_system = Guidelines()
    # Generate recommendations
    results = screening_system.generate_recommendations({"clinical_note" : clinical_note})
    
    print("\nRECOMMENDED SCREENING TESTS:")
    print(results["recommendations"])
    print("\n")

if __name__ == "__main__":
    main()