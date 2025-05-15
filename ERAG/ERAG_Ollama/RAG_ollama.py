import os
import gc
import time
import subprocess
import pandas as pd
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import faiss
import numpy as np

#  Docker constants
CONTAINER_NAME = "PRW_ollama"  # Your Docker container name
MODEL_NAME = "llama3.1:70b"  # LLaMa model inside Docker

#  Instructions for LLaMa model processing
summarization_instructions = """
You are a highly skilled pathologist assistant tasked with accurately identifying the morphology and topography of the primary tumor. 
Tumor morphology refers to the histological classification of the tumor, while topography specifies the anatomical site of the tumor. 
Your expertise in analyzing detailed medical reports will ensure precise identification of these attributes. " 
Summarised the givien pathology report based on fidning a single mosty seriosu tumor and its morphology and topogrpahy. 
 ** do not perform othewr task other than finidng morphology and topography of the most seriosu tumor**
"""

morphology_extraction_instructions = """
From the  given finding extract **only the morphology description** of the tumor.
Return just the morphology information, and nothing else.
"""

topography_extraction_instructions = """
From the summarized pathology report, extract **only the topography location** of the tumor.
Return just the topography information, and nothing else.
"""
#  New structured prompts for selecting the final SNOMED code
final_morphology_selection_instructions = """
You are an expert in pathology coding. Your task is to analyze the extracted **morphology description** of a tumor and compare it with three possible SNOMED morphology codes. 
Follow these steps:
1. **Understand** the provided morphology description.
2. **Compare** it with each of the given SNOMED morphology codes.
3. **Select the most accurate code** based on the closest match.

Only return your answer in this format:

The morphology is <Morphology> and the its SNOMED Code is <BEST_MCODE>
"""


final_topography_selection_instructions = """
You are an expert in pathology coding. Your task is to analyze the extracted **topography location** of a tumor and compare it with three possible SNOMED topography codes. 
Follow these steps:
1. **Understand** the provided topography description.
2. **Compare** it with each of the given SNOMED topography codes.
3. **Select the most accurate code** based on the closest match.

Only return your answer in this format:
The Topogrpahy is <Topography> and the its SNOMED Code is <BEST_TCODE>
"""

#  Normalize embeddings for cosine similarity
def normalize_embeddings(vectors):
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

#  Load CSV files into FAISS with Cosine Similarity
def load_and_index_csv(csv_path):
    """Load SNOMED CSV and index it using FAISS for similarity search."""
    loader = CSVLoader(file_path=csv_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Use FAISS properly with LangChain
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    return vectorstore




#  Load FAISS databases for RAG
topography_vectorstore = load_and_index_csv("Topography_SNOMED.csv")
morphology_vectorstore = load_and_index_csv("Morphology_SNOMED.csv")

#  Call LLaMa using subprocess inside Docker
def call_llama_subprocess(content, instructions):
    """Run LLaMa 3.1:70B inside Docker using subprocess"""
    complete_prompt = f"{instructions}\n{content}\n"
    try:
        process = subprocess.Popen(
            ["docker", "exec", "-i", CONTAINER_NAME, "ollama", "run", MODEL_NAME],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        output, error = process.communicate(input=complete_prompt)
        return output.strip() if output else error.strip()
    except Exception as e:
        return str(e)

#  RAG Query for Topography and Morphology
def rag_query(query, vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Retrieve top 3 results
    retrieved_docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    return context


def validate_topography_code(extracted_topography, retrieved_topography_codes):
    """Pass extracted topography and three retrieved SNOMED topography codes to LLaMa for final selection."""
    validation_prompt = f"""
    Extracted Topography: {extracted_topography}

    Possible SNOMED Topography Codes:
    {retrieved_topography_codes}

    {final_topography_selection_instructions}
    """
    return call_llama_subprocess(validation_prompt, final_topography_selection_instructions)

def validate_morphology_code(extracted_morphology, retrieved_morphology_codes):
    """Pass extracted morphology and three retrieved SNOMED morphology codes to LLaMa for final selection."""
    validation_prompt = f"""
    Extracted Morphology: {extracted_morphology}

    Possible SNOMED Morphology Codes:
    {retrieved_morphology_codes}

    {final_morphology_selection_instructions}
    """
    return call_llama_subprocess(validation_prompt, final_morphology_selection_instructions)
    
    
#  Main processing function
def main():
    while True:
        print("\nEnter your pathology report (or type 'exit' to quit):")
        user_input = input("> ").strip()

        if user_input.lower() == "exit":
            print("Exiting the program.")
            break

        #  Step 1: Summarization
        print("\nProcessing Summarization...")
        summary = call_llama_subprocess(user_input, summarization_instructions)
        if "Error" in summary:
            print(f"Summarization failed: {summary}")
            continue
        print("\nSummary:")
        print(summary)

        #  Step 2: Extract Morphology from Summary
        print("\nExtracting Morphology...")
        morphology_text = call_llama_subprocess(summary, morphology_extraction_instructions)
        if "Error" in morphology_text:
            print(f"Morphology extraction failed: {morphology_text}")
            continue
        print("\nExtracted Morphology:")
        print(morphology_text)

        #  Step 3: Extract Topography from Summary
        print("\nExtracting Topography...")
        topography_text = call_llama_subprocess(summary, topography_extraction_instructions)
        if "Error" in topography_text:
            print(f"Topography extraction failed: {topography_text}")
            continue
        print("\nExtracted Topography:")
        print(topography_text)

        #  Step 4: Retrieve Topography Code using RAG
        print("\nFinding Topography Code using RAG...")
        topography_result = rag_query(topography_text, topography_vectorstore)
        print("\nTopography Code retrieved:")
        print(topography_result)

        #  Step 5: Retrieve Morphology Code using RAG
        print("\nFinding Morphology Code using RAG...")
        morphology_result = rag_query(morphology_text, morphology_vectorstore)
        print("\nMorphology Code Retrieved:")
        print(morphology_result)
        
        #  Step 6: Finalize the Best Topography Code
        final_topography_code = validate_topography_code(topography_text, topography_result)
        print("\nFinal Topography Code:", final_topography_code)

        #  Step 7: Finalize the Best Morphology Code
        final_morphology_code = validate_morphology_code(morphology_text, morphology_result)
        print("\nFinal Morphology Code:", final_morphology_code)

        #  Free memory
        gc.collect()
        time.sleep(1)

if __name__ == "__main__":
    main()

