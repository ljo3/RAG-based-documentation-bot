import streamlit as st
import time
from dotenv import load_dotenv
from openai import OpenAI
import concurrent.futures

import nltk
try:
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    stopwords.words('english') # Accessing it once forces it to load fully
except:
    pass

load_dotenv()
model = "gpt-5-mini"
client = OpenAI()


SOURCE_FILE = "./data/RealCostOfHS2.txt"
SOURCE_FOLDER = "./data"
embed_model_name = "BAAI/bge-small-en-v1.5"

query = "What is HS2 and Why is it important?"

def rag_manual(source, query):
    from fastembed import TextEmbedding
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    with open(source, "r", encoding="UTF-8") as file:
        documents = file.read()

    # Embed using small model - 384 parameters
    embed_model = TextEmbedding(model_name=embed_model_name)
    doc_embeddings = list(embed_model.embed(documents)) # fastembed gives us generator, so convert to list

    # Embed query
    query_embedding = list(embed_model.embed([query]))[0]

    # Calculate cosine similarity
    scores = cosine_similarity([query_embedding], np.stack(doc_embeddings))[0]

    # Find the index of the highest score
    best_doc_index = np.argmax(scores)
    retrieved_doc = documents[best_doc_index]

    # print(f"Retrieved: {retrieved_doc} (Score: {scores[best_doc_index]:.4f})")
    prompt = f"Context: {retrieved_doc}\nQuestion: {query}\nAnswer:"
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def rag_llamaIndex_openai_embed(source_folder,query):
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

    # setup document for llamaindex
    documents = SimpleDirectoryReader(source_folder).load_data()

    # index
    index = VectorStoreIndex.from_documents(documents=documents)

    # retrieve and generate
    query_engine = index.as_query_engine()
    response = query_engine.query(query)

    return response

def rag_llamaIndex_local_embed(source_folder,query):
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
    from llama_index.embeddings.fastembed import FastEmbedEmbedding

    # configure LlamaIndex to use FastEmbed running locally
    Settings.embed_model = FastEmbedEmbedding(model_name=embed_model_name)

    # setup document for llamaindex
    documents = SimpleDirectoryReader(source_folder).load_data()

    # index
    index = VectorStoreIndex.from_documents(documents=documents)

    # retrieve and generate
    query_engine = index.as_query_engine()
    response = query_engine.query(query)

    return response

def rag_qdrant(source_folder, query):
    import qdrant_client
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    from llama_index.embeddings.fastembed import FastEmbedEmbedding
    from llama_index.llms.openai import OpenAI

    # Setup llm & embeddings
    Settings.llm = OpenAI(model=model)

    # Vector Store Qdrant
    client = qdrant_client.QdrantClient(location=":memory:")

    # storage
    vector_store = QdrantVectorStore(client=client, collection_name="HS2")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # index
    documents = SimpleDirectoryReader(source_folder).load_data()
    index = VectorStoreIndex.from_documents(documents=documents, storage_context=storage_context)

    # query
    query_engine = index.as_query_engine(similarity_top_k=1)
    response = query_engine.query(query)
    return response


def run_pipeline_wrapper(func, source_arg, query):
    start = time.time()
    try:
        # We pass source_arg (folder or file) AND query here
        response = func(source_arg, query)
        error = None
    except Exception as e:
        response = None
        error = str(e)
    end = time.time()
    return {"resp": response, "time": end - start, "error": error}

# --- UI CONFIGURATION ---
st.set_page_config(page_title="RAG Architecture Battle", layout="wide")

st.title("To Build or To Buy: RAG Architecture")
st.markdown("""
**Objective:** Compare a manual 'from scratch' RAG implementation against different configurations of optimized LlamaIndex pipeline.
*Use the sidebar to configure the data sources.*
""")

# Input
query_input = st.text_input("Enter your question:", value="What is HS2 and why is it important?")

if st.button("Run All Comparisons", type="primary"):
    
    # Create 4 Columns
    col1, col2, col3, col4 = st.columns(4)

    # Show loading spinners in all columns
    with col1: st.write("⏳ Running Manual...")
    with col2: st.write("⏳ Running OpenAI...")
    with col3: st.write("⏳ Running Local...")
    with col4: st.write("⏳ Running Qdrant...")    
    
    # Execute in Parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:

        # 1. Manual takes SOURCE_FILE
        future_manual = executor.submit(run_pipeline_wrapper, rag_manual, SOURCE_FILE, query_input)
        
        # 2, 3, 4 take SOURCE_FOLDER
        future_openai = executor.submit(run_pipeline_wrapper, rag_llamaIndex_openai_embed, SOURCE_FOLDER, query_input)
        future_local = executor.submit(run_pipeline_wrapper, rag_llamaIndex_local_embed, SOURCE_FOLDER, query_input)
        future_qdrant = executor.submit(run_pipeline_wrapper, rag_qdrant, SOURCE_FOLDER, query_input)        
        # Wait for all to complete
        res_manual = future_manual.result()
        res_openai = future_openai.result()
        res_local = future_local.result()
        res_qdrant = future_qdrant.result()

    # --- RENDER RESULTS (Main Thread) ---
    
    # Column 1: Manual
    with col1:
        st.subheader("1. Manual")
        if res_manual['error']:
            st.error(res_manual['error'])
        else:
            st.metric("Latency", f"{res_manual['time']:.2f}s")
            st.success(res_manual['resp'])

    # Column 2: OpenAI
    with col2:
        st.subheader("2. LlamaIndex + OpenAI Embedding")
        if res_openai['error']:
            st.error(res_openai['error'])
        else:
            st.metric("Latency", f"{res_openai['time']:.2f}s")
            st.success(res_openai['resp'])

    # Column 3: Local
    with col3:
        st.subheader("3. LlamaIndex + Local Embedding")
        if res_local['error']:
            st.error(res_local['error'])
        else:
            st.metric("Latency", f"{res_local['time']:.2f}s")
            st.success(res_local['resp'])

    # Column 4: Qdrant
    with col4:
        st.subheader("4. LlamaIndex + Qdrant")
        if res_qdrant['error']:
            st.error(res_qdrant['error'])
        else:
            st.metric("Latency", f"{res_qdrant['time']:.2f}s")
            st.success(res_qdrant['resp'])
