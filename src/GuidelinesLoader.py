from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
import os
import glob
from typing import List, Dict, Any
import json
# from langchain_community.document_loaders import PDFLoader
from langchain_community.document_loaders import PyPDFLoader

class GuidelinesLoader(BaseLoader):
    """Loader for medical guidelines that uses an LLM to extract metadata."""
    
    def __init__(self, directory_path: str, llm, glob_pattern: str = "**/*.pdf"):
        """Initialize with directory path, LLM model, and glob pattern."""
        self.directory_path = directory_path
        self.glob_pattern = glob_pattern
        self.llm = llm
        
    def load(self) -> List[Document]:
        """Load documents and extract metadata using LLM."""
        guideline_files = glob.glob(
            os.path.join(self.directory_path, self.glob_pattern), 
            recursive=True
        )
        
        documents = []
        for file_path in guideline_files:
            print(f"Processing {file_path}...")
            
            # Load the content
            # text_loader = TextLoader(file_path)
            text_loader = PyPDFLoader(file_path)
            file_docs = text_loader.load()
            
            if not file_docs:
                continue
                
            # The content of the document
            content = file_docs[0].page_content
            filename = os.path.basename(file_path)
            
            # Extract metadata using LLM
            metadata = self._extract_metadata_with_llm(content, filename, file_path)
            
            # Create a new document with the content and metadata
            documents.append(
                Document(
                    page_content=content,
                    metadata=metadata
                )
            )
        
        return documents
    
    def _extract_metadata_with_llm(self, content: str, filename: str, file_path: str) -> Dict[str, Any]:
        """Extract metadata from document content using an LLM."""
        # Create a prompt for the LLM to extract metadata
        prompt = f"""
        Extract metadata from this medical guideline document as structured JSON.
        
        Filename: {filename}
        
        First 3000 characters of document content: 
        {content[:3000]}...
        
        Extract the following metadata:
        1. governing_body: The organization or professional society that published this guideline (e.g., USPSTF, IDSA, AMA). Name the same governing_body the same way every time. 
        2. topic: The medical condition or topic this guideline addresses
        3. pub_date: The full date if available (YYYY-MM-DD format)
        4. evidence_grade: The strength of evidence is given a grade of one or more of the following choices: A, B, C, D, I
        5. population.gender: Gender this applies to. Encode 1 of the following 3 choices: male, female, both
        6. population.min_age: Minimum age this applies to (number)
        7. population.max_age: Maximum age this applies to (number)
        8. population.pregnant: Whether this applies to pregnant women (true/false)
        9. screening_interval: How frequent screening should occur, in years. That is, the patient should be screened every X year(s).
        10. type: Type of guideline (e.g., Screening, Prevention, Treatment)

        The metadata is often, but not always, found in a summary section at the beginning of the document in a section
        which has the following format:

        IMPORTANCE <the importance of the guideline>
        OBJECTIVE <the objective of the guideline>
        POPULATION <the age group, gender, and criteria for whom to apply the guideline to>
        EVIDENCE ASSESSMENT <the evidence supporting the recommendation>
        RECOMMENDATION <the recommendations(s) themselves>

        If you can not determine the metadata from this summary section, keep looking through the rest of the document.
        
        Format your response as a valid JSON object containing ONLY these fields. Do not nest any fields.
        If a metadata field cannot be determined from the document's content, keep the key but make its value null.
        Do not make up information - only extract what is explicitly stated or can be reasonably inferred.
        """
        
        # Call the LLM to extract metadata
        response = self.llm.invoke(prompt)
        print(f"LLM response: {response.content}")
        
        try:
            # Parse the LLM's response as JSON
            # We need to extract just the JSON portion from the response
            json_str = self._extract_json_from_response(response.content)
            # print(f"Extracted JSON string: {json_str}")
            metadata = json.loads(json_str)
            print(f"Parsed metadata: {metadata}")
            
            # Add source file path (not from LLM)
            metadata["source"] = file_path
            
            # Ensure numeric fields are proper numbers
            if "population.min_age" in metadata and metadata["population.min_age"] is not None:
                metadata["population.min_age"] = int(metadata["population.min_age"])
            if "population.max_age" in metadata and metadata["population.max_age"] is not None:
                metadata["population.max_age"] = int(metadata["population.max_age"])
            if "screening_interval" in metadata and metadata["screening_interval"] is not None:
                metadata["screening_interval"] = int(metadata["screening_interval"])

            #Chroma can't accept lists nor None values: convert lists to strings; None to empty string ""
            for key, value in metadata.items():
                if isinstance(value, list):
                    metadata[key] = ", ".join(map(str, value))
                elif value is None:
                    metadata[key] = ""
                
            return metadata
            
        except Exception as e:
            print(f"Error extracting metadata with LLM: {e}")
            # Return basic metadata if LLM extraction fails
            return {
                "source": file_path,
                "organization": "Unknown",
                "condition": os.path.basename(os.path.dirname(file_path)),
                "extraction_error": str(e)
            }
    
    def _extract_json_from_response(self, response: str) -> str:
        """Extract JSON string from LLM response, handling various response formats."""
        # Try to find JSON between triple backticks
        import re
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
        if json_match:
            return json_match.group(1)
        
        # Try to find JSON between curly braces
        json_match = re.search(r'(\{[\s\S]*\})', response)
        if json_match:
            return json_match.group(1)
        
        # If no JSON format found, return the full response
        return response







