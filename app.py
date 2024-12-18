from transformers import pipeline
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from Embedding import Embedding
import faiss
import rouge
import streamlit as st
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
st.set_page_config(layout="wide")

# Streamlit UI
st.title("Sample RAG Demo")
data = st.text_area("Input Data:")
chunk_size = st.number_input("Chunk Size", min_value=10, max_value=500, value=50, step=50)
overlap_width = st.number_input("Overlap Size", min_value=10, max_value=500, value=10, step=10)
query = st.text_input("Ask a question:")
Num_retrieve = st.number_input("Number of Retrievals", min_value=1, max_value=5, value=3, step=1)
temperature = st.slider("Temperature", min_value=0.0, max_value=1.5, value=0.7, step=0.1)

if st.button('Do RAG'):
    embed = Embedding()
    st.write("Data length:",len(data.split()))
    st.write("Data Loaded....")
    st.write("-----------------------------------------------------------------------")
    ##Normalizing
    #simple_text = embed.simple_normalize(data)
    #st.write("Simple Normalize:",simple_text)
    #st.write("Simple Length:",len(simple_text.split()))
    lemmatized,enhanced_real = embed.spacy_preprocess(data)
    #st.write("lemmatized Normalize:",lemmatized)
    #st.write("lemmatized Length:",len(lemmatized.split()))
    st.write("Enhanced Normalize:",enhanced_real)
    st.write("Enhanced Length:",len(enhanced_real.split()))
    st.write("Normalizing Finished....")
    st.write("-----------------------------------------------------------------------")
    ##Chuncking
    chunked = embed.chunk_text(enhanced_real,chunk_size,overlap_width)
    st.write("Chunking:",chunked)
    st.write("Chunking depth:",len(chunked),"Chunk size:",chunk_size,"Overlap width:",overlap_width)
    with open('my_chunks.txt', 'w',encoding='utf-8') as f:
        for c in chunked:
            f.write(f"{c}\n")
    st.write("Chunked File Written....")
    st.write("-----------------------------------------------------------------------")

    ##Embedding
    embeddings = embed.get_bert_embeddings(chunked)
    st.write("Embeddings Shape:",embeddings.shape)
    with open('my_embeddings.txt', 'w',encoding='utf-8') as f:
        for e in embeddings:
            f.write(f"{e}\n")
    st.write("Embeddings File Written....")
    st.write("-----------------------------------------------------------------------")

    ##Indexing
    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance for similarity search
    faiss_index.add(embeddings)
    faiss.write_index(faiss_index, "faiss_index.bin")
    st.write("Fais Index File Written....")
    st.write("-----------------------------------------------------------------------")

    ##Adjust Number of relevant chunks and input query
    query_embedding = embed.get_bert_embeddings(query.lower())

    ##Fais Reading and Search
    faiss_index = faiss.read_index("faiss_index.bin")
    distances, indices = faiss_index.search(query_embedding, Num_retrieve)
    faiss_results,scores = [],[]
    st.write(distances)
    for i, idx in enumerate(indices[0]):
        faiss_results.append(chunked[idx])
        scores.append(float(distances[0][i]))
        st.write("Result",i,":",chunked[idx])
        st.write("Score:",float(distances[0][i]))
    st.write("Retrieved Done For best related",Num_retrieve," Answers using FAISS....")
    st.write("-----------------------------------------------------------------------")

    ## Cosine Similarity Search and Retrieving
    similarities = cosine_similarity(query_embedding, embeddings)
    st.write("Cosine Similarities:",similarities)
    top_k_idx = np.argsort(similarities[0])[-Num_retrieve:][::-1]
    cosine_results = [chunked[i] for i in top_k_idx]
    for i, idx in enumerate(cosine_results):
        st.write("Result",i,":",cosine_results[i])
        st.write("Score:",float(similarities[0][i]))
    st.write("Retrieved Done For best related",Num_retrieve," Answers using Cosine Similarity....")
    st.write("-----------------------------------------------------------------------")


    ##Generating Answer form LLM with retrieved Context


    LLM_Model = 'gpt2'
    llm = pipeline(task='text-generation', model=LLM_Model)
    llm.model.config.pad_token_id = llm.model.config.eos_token_id
    generated = llm(f"Query: {query}\nContext: {' '.join(cosine_results)}\nAnswer:",max_new_tokens=150,temperature=temperature,num_return_sequences=1)

    answer = generated[0]['generated_text'].split('Answer:')[1]
    st.write("Answer on Faiss related retrieved context:", answer)
    evaluator = rouge.Rouge()
    rouge_cosine = evaluator.get_scores(answer, ' '.join(cosine_results))
    st.write("Rouge Score:",rouge_cosine)
    st.write("Generated Answers From LLM using Retrieved Cosine Similarities Context with Temperature",temperature,"....")
    st.write("-----------------------------------------------------------------------")

    generated = llm(f"Query: {query}\nContext: {' '.join(faiss_results)}\nAnswer:",max_new_tokens=150,temperature=temperature,num_return_sequences=1)

    answer = generated[0]['generated_text'].split('Answer:')[1]
    st.write("Answer on Cosine related retrieved context:", answer)
    rouge_cosine = evaluator.get_scores(answer, ' '.join(faiss_results))
    st.write("Rouge Score:",rouge_cosine)
    st.write("Generated Answers From LLM using Retrieved FAIS Index Context with Temperature",temperature,"....")
    st.write("-----------------------------------------------------------------------")