import streamlit as st
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains import RetrievalQA
import spacy
from spacy import displacy
import es_core_news_sm

def NER_Model(user_input):
    list_ner=[]
    nlp = es_core_news_sm.load()

    #NER = spacy.load("es_core_news_lg")
    text1 = nlp(user_input)
    print(text1.ents)
    for word in text1.ents:
        list_ner.append((word.text, word.label_))
    return {'ner': list_ner}

def LLM_Model(user_input):
    loader = CSVLoader(file_path='productos.csv')
    docs = loader.load()
    embeddings = HuggingFaceEmbeddings()
    vector_store = FAISS.from_documents(docs, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    llm = OllamaLLM(model="deepseek-r1:1.5b")
    prompt = """
        1. Use ONLY the context below.
        2. If unsure, say "I don’t know".
        3. Eres un asistente en CRM

        Context: {context}

        Question: {question}

        Answer:
        """
    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)
    llm_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT)

    # Combine document chunks
    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=QA_CHAIN_PROMPT
    )
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
    )
    return qa.run(user_input), NER_Model(user_input)
