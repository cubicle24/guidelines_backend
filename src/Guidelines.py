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
from langchain_fireworks import ChatFireworks
import shutil



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
        self.choose_model("google")
        # self.embeddings = SentenceTransformerEmbeddings('all-MiniLM-L6-v2')
        self.setup_vector_db()
        self.setup_rag_pipeline()

    def choose_model(self, model_name):
        """Set up the model and embeddings based on the model name. Allows you to switch LLMs"""
        if model_name == "google":
            print(f"selecting google model")
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", temperature=0.0)
            # self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=os.environ.get("GOOGLE_API_KEY"))
            self.embeddings = SentenceTransformerEmbeddings('emilyalsentzer/Bio_ClinicalBERT')

        elif model_name == "deepseek":
            print(f"selecting deepseek model")
            self.llm = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0.0")
            self.embeddings = SentenceTransformerEmbeddings('emilyalsentzer/Bio_ClinicalBERT')
        elif model_name == "llama3":
            print(f"selecting llama3 model")
            self.llm = ChatGroq(model="llama3-70b-8192", temperature=0.0")
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
    # Remove the existing database directory if it exists

            print("Creating new guidelines database")
            
            loader = GuidelinesLoader("../guidelines_repository", llm=self.llm)
            documents = loader.load()
            
            print(f"Loaded {len(documents)} guideline documents and extracted metadata")
            
            # Split documents while preserving metadata
            text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ".", " "],
            chunk_size=1000, chunk_overlap=100)
            splits = text_splitter.split_documents(documents)
            # splits = documents #skips chunking
            
            print(f"Split into {len(splits)} chunks for indexing")
            
            # Create vector database
            self.vector_db = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
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
            You are a backend API response generator. Output must be valid JSON only. Do not include any preamble, commentary, or prose. Only return a single JSON object that matches the format provided.

            Your task is to generate evidence-based screening recommendations for a patient based on their medical information and the provided clinical guidelines.
            First, extract the patient's age, gender, and pregnancy status. These 3 pieces of information are the most important for determining which screening tests to recommend.

            Recommend all screening tests that the patient meets the criteria for, based primarily on the patient's age, gender, and pregnancy status. 
            Never recommend a screening test for a male that only applies to females, and vice versa.  If a female is not pregnant, do not recommend any pregnancy specific tests.

            Do not make up information.  Do not query the internet.  Do not use your general knowledge. Only conssult the retrieved documents. 
            If a test is not indicated, do not mention or recommend it at all. Double check your recommendations to never make the error of recommending a test that the patient does not meet criteria for.

            If you can not make any recommendations, state that you can not make any recommendations.
            
            Generate a ranked list of recommended screening tests from highest to lowest priority.  
            For each recommendation, provide your justification and reasoning for the recommendation based on the guidelines.  Limit the justification to 100 words or less.            
            If your justification is that the patient does not fit the guidelines and are therefore not recommending a test, remove the recommendation from the list.  Do not list it all.

            Then cite the evidence (source) by finding the metadata associated with the retrieved documents. Cite the "governing_body", "topic", and "pub_date" that you consulted to make the recommendation.
            
            After first looking only at the provided guidelines and giving recommendations, attach a set of additional recommendations.
            For this section, read the entire clinical note, then you may use your general knowledge as well as ONLY search the US Preventive Task 
            Force Guidelines website for additional screening tests that the patient meets criteria for, focusing on age, gender, and pregnancy status. 
            Make as many additional recommendations as you can find reasonable justification for. Attach these
            additional recommendations under the JSON key "additional_recommendations" so that it is clear these come 
            from outside sources.

            Patient Information:
            {patient_data}
            
            Relevant Guidelines:
            {guidelines}

            If you have no guidelines to consult, return an empty but valid JSON object. 
            Return only the JSON object, with no extra text or formatting.
            If you have relevant guidelines to consult, order recommendations as a ranked list (from highest to lowest priority).  
            Return all recommendations in this EXACT JSON structure and format below. 
            Do not include any text outside the JSON object. Double check five times and self-correct that valid JSON is returned.
            Before you return the JSON object, triple check that the JSON follows the structure below and modify the structure if it does not.
           

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
                        "pub_date": "Not specified"
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
            search_kwargs={"k":10},search_type="mmr",lambda_mult=0.1)
            # search_kwargs={"k":15},search_type="similarity")
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

        #original
        # def retrieve_and_format_guidelines(inputs):
        #     # relevant_docs = retriever.invoke(query)
        #     relevant_docs = retriever.invoke(inputs.get("clinical_note",""))
        #     # for doc in relevant_docs:
        #     #     print(f"doc: {doc}\n")
        #     return {
        #         "patient_data": inputs["patient_data"],
        #         "guidelines": "\n\n".join([doc.page_content for doc in relevant_docs])
        #     }

        # def retrieve_and_format_guidelines(inputs):
        #     print(f'inputs: {inputs}')
        #     query = inputs.get("clinical_note", "")
        #     print("\n==== RETRIEVAL DEBUG ====")
        #     print(f"Query sent to retriever:\n{query}\n")
        #     relevant_docs = retriever.invoke(query)
        #     print(f"Number of retrieved docs: {len(relevant_docs)}")
        #     for i, doc in enumerate(relevant_docs):
        #         print(f"\n--- Retrieved Doc #{i+1} ---")
        #         # print(doc.page_content[:2500])  # Print first 500 chars for brevity
        #         print(doc.page_content)  # Print first 500 chars for brevity
        #     print("=========================\n")
        #     return {
        #         "patient_data": inputs["patient_data"],
        #         "guidelines": "\n\n".join([doc.page_content for doc in relevant_docs])
        #     }

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
            # {"clinical_note": lambda x: x}
            # Extract patient information using JSON output parser
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
        # Print intermediate results for debugging
        # print(f"Extracted patient data: {result['patient_data']}")
        
        return {
            "patient_data": result["patient_data"],
            # "recommendations": result["recommendations"].content
            "recommendations": result["screening_recommendations"]
        }

def main():
    # Initialize the system - no API key needed as we're using GOOGLE_API_KEY from environment
    screening_system = Guidelines()
    # Generate recommendations
    results = screening_system.generate_recommendations({"clinical_note" : clinical_note})
    
    # Print recommendations
    print("\nRECOMMENDED SCREENING TESTS:")
    print(results["recommendations"])
    print("\n")

if __name__ == "__main__":
    main()