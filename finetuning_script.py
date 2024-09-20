"""
finetuning_script.py

This python script is set up to run just the finetuning portion of the AIE4 Midterm assignment...

NOTE: All other midterm steps are run via a separate notebook...
I set this up as a separate script as I encountered some issues when running the finetuning portion
of the notebook on Google Colab.

This separate python script will only focus on the finetuning portion of the assignment and will run
on AWS EC2...I will push the HF model from there to the hub and use it in the notebook to work
on all the other steps for the midterm...

"""

import os
import dotenv

# from operator import itemgetter
# import pandas as pd
# from typing import List

# from langchain.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.documents import Document

# from langchain_community.document_loaders import PyMuPDFLoader

# from datasets import Dataset

# from ragas import evaluate
# from ragas.metrics import faithfulness, answer_relevancy, answer_correctness, context_recall, context_precision
# from ragas.testset.evolutions import simple, reasoning, multi_context

from myutils.rag_pipeline_utils import SimpleTextSplitter, SemanticTextSplitter, VectorStore, AdvancedRetriever
# from myutils.ragas_pipeline import RagasPipeline

from sentence_transformers import SentenceTransformer

# from torch.utils.data import DataLoader
# from torch.utils.data import Dataset
# from sentence_transformers import InputExample
# from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
# from sentence_transformers.evaluation import InformationRetrievalEvaluator
from langchain_huggingface import HuggingFaceEmbeddings
# import pandas as pd

# from langchain_community.vectorstores import FAISS
# from langchain_openai.embeddings import OpenAIEmbeddings
# from langchain_core.documents import Document

from datetime import datetime
import time

import nest_asyncio

nest_asyncio.apply()


# my module imports
# import sys and point to top-level location
# this allows the module path references to stay the same for runs from command line
import sys
sys.path.append('./')

from myutils.rag_pipeline_utils import load_all_pdfs
from myutils.finetuning import PrepareDataForFinetuning, FineTuneModelAndEvaluateRetriever

import logging


dotenv.load_dotenv()

pdf_file_paths = [
    './data/docs_for_rag/Blueprint-for-an-AI-Bill-of-Rights.pdf',
    './data/docs_for_rag/NIST.AI.600-1.pdf'
]


class FineTuneEmbeddingModel:
    def __init__(self,
                 pdf_file_paths=pdf_file_paths,
                 chunk_size=1000,
                 chunk_overlap=300,
                 train_val_test_fraction=[0.80, 0.10, 0.10],
                 train_val_test_split_type='random',
                 qa_chat_model_name='gpt-4o-mini',
                 random_seed=69,
                 n_questions=3,
                 batch_size=64,
                 base_model_id='Snowflake/snowflake-arctic-embed-m',
                 matryoshka_dimensions=[768, 512, 256, 128, 64],
                 number_of_training_epochs=5,
                 finetuned_model_output_path='finetuned_arctic',
                 evaluation_steps=50,
                 hf_repo="vincha77/finetuned_arctic"):
        
        self.pdf_file_paths = pdf_file_paths
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.train_val_test_fraction = train_val_test_fraction
        self.train_val_test_split_type = train_val_test_split_type

        self.qa_chat_model_name = qa_chat_model_name
        
        self.random_seed = random_seed
        self.n_questions = n_questions
        self.batch_size = batch_size

        self.base_model_id = base_model_id
        self.matryoshka_dimensions = matryoshka_dimensions
        self.number_of_training_epochs = number_of_training_epochs

        self.finetuned_model_output_path = finetuned_model_output_path
        self.evaluation_steps = evaluation_steps

        self.hf_repo = hf_repo
        return
    
    def load_and_chunk_docs(self):
        self.documents = load_all_pdfs(self.pdf_file_paths)
        # instantiate baseline text splitter -
        # NOTE!!! The `SimpleTextSplitter` below is my wrapper around Langchain RecursiveCharacterTextSplitter!!!!
        # (see module for the code if needed)
        baseline_text_splitter = SimpleTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap, 
            documents=self.documents
        )

        # split text for baseline case
        self.baseline_text_splits = baseline_text_splitter.split_text()
        return self
    
    def prep_data_for_finetuning(self):
        self.pdft = PrepareDataForFinetuning(
            all_splits=self.baseline_text_splits,
            train_val_test_fraction=self.train_val_test_fraction,
            train_val_test_split_type=self.train_val_test_split_type,
            random_seed=self.random_seed,
            qa_chat_model_name=self.qa_chat_model_name,
            n_questions=self.n_questions,
            batch_size=self.batch_size
        )

        self.pdft.run_all_prep_data()
        return self
    
    def finetune_and_eval_retriever(self):
        self.evr = FineTuneModelAndEvaluateRetriever(
            train_data=self.pdft.train_dataset,
            val_data=self.pdft.val_dataset,
            test_data=self.pdft.test_dataset,
            batch_size=self.batch_size,
            base_model_id=self.base_model_id,
            matryoshka_dimensions=self.matryoshka_dimensions,
            number_of_training_epochs=self.number_of_training_epochs,
            finetuned_model_output_path=self.finetuned_model_output_path,
            evaluation_steps=self.evaluation_steps
        )

        self.evr.run_steps_to_finetune_model()

        self.arctic_finetuned_model = SentenceTransformer(self.finetuned_model_output_path)
        return self
    
    def push_finetuned_model_to_hf_hub(self):
        self.arctic_finetuned_model.push_to_hub(self.hf_repo)
        return self
    
    def pull_finetuned_model_from_hf_hub(self):
        ## code here to pull from hub
        model_id = self.hf_repo
        arctic_finetuned_model = SentenceTransformer(model_id)
        arctic_finetuned_embeddings = HuggingFaceEmbeddings(model_name=self.hf_repo)
        return self
    
    def run_finetuning_steps(self):
        self.load_and_chunk_docs()
        self.prep_data_for_finetuning()
        self.finetune_and_eval_retriever()
        self.push_finetuned_model_to_hf_hub()
        self.pull_finetuned_model_from_hf_hub()
        return self


def main(runymd):
    ftem = FineTuneEmbeddingModel()
    ftem.run_finetuning_steps()
    return


if __name__ == "__main__":
    # set up log file details
    this_file = os.path.basename(__file__).replace('.py', '')
    log_id = f'{this_file}_{datetime.now().strftime("%Y%m%dT%H%M%S")}'
    logfiledir = './logs'
    logfilename = f'{log_id}.log'
    logfilepath = f'{logfiledir}/{logfilename}'
    
    # set up logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=f'{logfilepath}', level=logging.INFO,
                        format=f'%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    start_time = time.time()
    runymd = datetime.today().strftime('%Y-%m-%d')
    logger.info(f'starting the code on {runymd} ')
    main(runymd)
    end_time = time.time()
    logger.info('time taken to run program in %s seconds:' % (end_time - start_time))
    logger.info(f'successfully completed script')
