"""
app_end_to_end_prototype.py

1. This app loads two pdf documents and allows the user to ask questions about these documents.
    The documents that are used are:
    
    https://www.whitehouse.gov/wp-content/uploads/2022/10/Blueprint-for-an-AI-Bill-of-Rights.pdf
    AND
    https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf

2. The two documents are pre-processed on start.  Here are brief details on the pre-processing:
    a.  text is split into chunks using langchain RecursiveCharacterTextSplitter method.
    b.  The text in each chunk is converted to an embedding using OpenAI text-embedding-3-small embeddings.
        Each embedding produced by this model has dimension 1536.
        Each chunk is therefore represented by an embedding of dimension 1536.
    c.  The collection of embeddings for all chunks along with metadata are saved/indexed in a vector database.
    d.  For this exercise, I use an in-memory version of Qdrant vector db.

3.  The next step is to build a RAG pipeline to answer questions.  This is implemented as follows:
    a.  I use a simple prompt that retrieves relevant contexts based on a user query.
    b.  First, the user query is encoded using the same embedding model as the documents.
    c.  Second, a set of relevant documents is returned by the retriever 
        which efficiently searches the vector db and returns the most relevant chunks.
    d.  Third, the user query and retrieved contexts are then passed to a chat-enabled LLM.
        I use OpenAI's gpt-4o-mini throughout this exercise.
    e.  Fourth, the chat model processes the user query and context along with the prompt and 
        generates a response that is then passed to the user.

4.  The cl.on_start initiates the conversation with the user.

5.  The cl.on_message decorator wraps the main function
        This function does the following:
            a. receives the query that the user types in
            b. runs the RAG pipeline
            c. sends results back to UI for display

Additional Notes:
a. note the use of async functions and await async syntax throughout the module here!
b. note the use of yield rather than return in certain key functions
c. note the use of streaming capabilities when needed

"""

import os
from typing import List
from dotenv import load_dotenv

# chainlit imports
import chainlit as cl

# langchain imports
# document loader
from langchain_community.document_loaders import PyPDFLoader
# text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
# embeddings model to embed each chunk of text in doc
from langchain_openai import OpenAIEmbeddings
# vector store
# llm for text generation using prompt plus retrieved context plus query
from langchain_openai import ChatOpenAI
# templates to create custom prompts
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
# chains 
# LCEL Runnable Passthrough
from langchain_core.runnables import RunnablePassthrough
# to parse output from llm
from langchain_core.output_parsers import StrOutputParser
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import PyMuPDFLoader

from sentence_transformers import SentenceTransformer

from myutils.rag_pipeline_utils import SimpleTextSplitter, SemanticTextSplitter, VectorStore, AdvancedRetriever
from myutils.ragas_pipeline import RagasPipeline
from myutils.rag_pipeline_utils import load_all_pdfs, set_up_rag_pipeline


load_dotenv()

# Flag to indicate if pdfs should be loaded directly from URLs
# If True, get pdfs from urls; if false, get them from local copy
LOAD_PDF_DIRECTLY_FROM_URL = True

# set the APP_MODE
# one of two choices:
# early_prototype means use OpenAI embeddings
# advanced_prototype means use finetuned model embeddings
APP_MODE = "early_prototype"

if APP_MODE == "advanced_prototype":
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
    embed_dim = 1536
    appendix_to_user_message = "This chatbot is built using OpenAI Embeddings as a fast prototype."
else:
    finetuned_model_id = "vincha77/finetuned_arctic"
    arctic_finetuned_model = SentenceTransformer(finetuned_model_id)
    embeddings = HuggingFaceEmbeddings(model_name="vincha77/finetuned_arctic")
    appendix_to_user_message = "Our Tech team finetuned snowflake-arctic-embed-m to bring you this chatbot!!"
    embed_dim = 768

rag_template = """
You are an assistant for question-answering tasks.
You will be given documents on the risks of AI, frameworks and
policies formulated by various governmental agencies to articulate
these risks and to safeguard against these risks.

Use the following pieces of retrieved context to answer 
the question. 

You must answer the question only based on the context provided.

If you don't know the answer or if the context does not provide sufficient information, 
then say that you don't know. 

Think through your answer step-by-step.

Context:
{context}

Question: 
{question}
"""

rag_prompt = ChatPromptTemplate.from_template(template=rag_template)

# parameters to manage text splitting/chunking
chunk_kwargs = {
    'chunk_size': 1000,
    'chunk_overlap': 300
}

retrieval_chain_kwargs = {
    'location': ":memory:",
    'collection_name': 'End_to_End_Prototype',
    'embeddings': embeddings,
    'embed_dim': embed_dim,
    'prompt': rag_prompt,
    'qa_llm': ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
}

urls_for_pdfs = [
    "https://www.whitehouse.gov/wp-content/uploads/2022/10/Blueprint-for-an-AI-Bill-of-Rights.pdf",
    "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf"
]

pdf_file_paths = [
    './data/docs_for_rag/Blueprint-for-an-AI-Bill-of-Rights.pdf',
    './data/docs_for_rag/NIST.AI.600-1.pdf'
]

# if flag is True, then pass in pointers to URLs
# if flag is false, then pass in file pointers
if LOAD_PDF_DIRECTLY_FROM_URL:
    docpathlist = urls_for_pdfs
else:
    docpathlist = pdf_file_paths


class RetrievalAugmentedQAPipelineWithLangchain:
    def __init__(self, 
                 list_of_documents,
                 chunk_kwargs,
                 retrieval_chain_kwargs):
        self.list_of_documents = list_of_documents
        self.chunk_kwargs = chunk_kwargs
        self.retrieval_chain_kwargs = retrieval_chain_kwargs

        self.load_documents()
        self.split_text()
        self.set_up_rag_pipeline()
        return
    
    def load_documents(self):
        self.documents = load_all_pdfs(self.list_of_documents)
        return self
    
    def split_text(self):
        baseline_text_splitter = \
            SimpleTextSplitter(**self.chunk_kwargs, documents=self.documents)
        # split text for baseline case
        self.baseline_text_splits = baseline_text_splitter.split_text()
        return self
    
    def set_up_rag_pipeline(self):
        self.retrieval_chain = set_up_rag_pipeline(
            **self.retrieval_chain_kwargs,
            text_splits=self.baseline_text_splits
        )
        return self
    

RETRIEVAL_CHAIN = \
    RetrievalAugmentedQAPipelineWithLangchain(
        list_of_documents=docpathlist,
        chunk_kwargs=chunk_kwargs,
        retrieval_chain_kwargs=retrieval_chain_kwargs
    ).retrieval_chain


@cl.on_chat_start
async def on_chat_start():

    msg = cl.Message(content=f"""
                    Hello dear colleague!  Welcome to this chatbot!  In recent weeks, many of you have shared that you'd like to understand how AI is evolving.  What better way to help you understand the implications of AI than "use AI to answer questions about AI".  Your colleagues in Technology have worked hard to create this chatbot.  We've used a few key policy and framework proposals from the US government that this chatbot can search for a response to your question.  Occasionally, the chatbot may respond with "I don't know".  If it does that, try a more specific variation of your question.  Oh! And one more thing: {appendix_to_user_message}...Please go ahead and enter your question...
                    """
                    )

    await msg.send()
    cl.user_session.set("retrieval_chain", RETRIEVAL_CHAIN)

@cl.on_message
async def main(message):
    retrieval_chain = cl.user_session.get("retrieval_chain")

    msg = cl.Message(content="")

    # result = await raqa_chain.invoke({"input": message.content})
    result = await cl.make_async(retrieval_chain.invoke)({"question": message.content})

    # async for stream_resp in result["answer"]:
    for stream_resp in result["response"].content:
        await msg.stream_token(stream_resp)

    await msg.send()
