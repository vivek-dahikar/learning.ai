import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM as Ollama
from langchain.text_splitter import CharacterTextSplitter
import PyPDF2

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def split_text_into_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def generate_embeddings(chunks):
    embeddings_model = OllamaEmbeddings(model="llama3.2")
    return FAISS.from_texts(chunks, embeddings_model)

def answer_query(query, vector_db):
    query_embedding = OllamaEmbeddings(model="llama3.2").embed_query(query)
    results = vector_db.similarity_search_by_vector(query_embedding, k=3)

    # Combine results into context for the LLM
    context = "\\n".join([result.page_content for result in results])
    llm = Ollama(model="llama3.2")
    answer = llm.generate([f"Context: {context}\nQuestion: {query}"])

    return answer.generations[0][0].text



def main():
    pdf_path = ""  # Replace with your PDF file path
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    text = extract_text_from_pdf(pdf_path)
    chunks = split_text_into_chunks(text)
  
    vector_db = generate_embeddings(chunks)

    while True:
        query = input("Ask a question (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        answer = answer_query(query, vector_db)
        print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
