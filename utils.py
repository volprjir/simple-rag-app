import os

from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import EMBEDDING_MODEL, OPENAI_MODEL

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")


def load_knowledge_base() -> FAISS:
    """
    Load the generated knowledge base
    :return:
        FAISS object
    """
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model=EMBEDDING_MODEL)
    DB_FAISS_PATH = os.path.join("document_preparation", "db_faiss")
    return FAISS.load_local(
        DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True
    )


def load_llm() -> ChatOpenAI:
    """
    Load the OpenAI LLM
    :return:
        ChatOpenAI LLM instance
    """
    llm = ChatOpenAI(model_name=OPENAI_MODEL, temperature=0, api_key=OPENAI_API_KEY)
    return llm


def load_prompt() -> ChatPromptTemplate:
    """
    Load the prompt template
    :return:
        ChatPromptTemplate instance
    """
    prompt = """ You need to answer the question in the sentence as same as in the text content. 
    Given below is the context and question of the user.
    context = {context}
    question = {question}
    if the answer is not in the context answer "I do apologize, but I am not able to provide an answer to that question at the moment."
     """
    prompt = ChatPromptTemplate.from_template(prompt)
    return prompt


def process_question(user_input: str) -> str:
    """
    Process the user input
    :param user_input: string received from the user
    :return:
        Generated response by the model
    """

    knowledge_base = load_knowledge_base()
    llm = load_llm()
    prompt = load_prompt()
    similar_embeddings = knowledge_base.similarity_search(user_input)
    similar_embeddings = FAISS.from_documents(
        documents=similar_embeddings,
        embedding=OpenAIEmbeddings(api_key=OPENAI_API_KEY, model=EMBEDDING_MODEL),
    )

    # creating the chain for integrating llm,prompt,stroutputparser
    retriever = similar_embeddings.as_retriever()
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke(user_input)
