#Takes a clinical note and generates a valid FHIR bundle of the most critical pieces

import re
from chromadb import api
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnableSequence  # Import RunnableSequence
import os
from langchain_core.output_parsers import JsonOutputParser
import shutil
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

def setup():
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not found.")
    self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", temperature=0.0, google_api_key=google_api_key)
