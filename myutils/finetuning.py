"""
finetuning_pipeline.py

Collects a number of methods in classes to streamline the finetuning of model embeddings


#### Fine-tuning Steps

1.  Prepare Train, Val and Test Data
    -   if needed, chunk data to get a list of LC Documents
    -   Split the list into train, val and test sub-groups
    -   For each sub-group, use an LLM to generate a list of POSITIVE question, context pairs. 
        -   This is done by passing the context to the LLM along with a prompt to generate `n_questions` number of questions; the questions are extracted from the LLM output and paired with the underlying context.  Note that each context will have more than one question paired with it.
    -   Write out the list of question, context pairs for train, val and test sub-groups into a jsonl file for future reference.
    -   The train sub-group is loaded into a HF Dataset object for use in training.
2.  Data Loader
    -   Set up data loader
    -   This includes the training data along with batch size information.
3.  Load model to be finetuned
    -   Use HF model name to load model
4.  Set up loss function
    -   concept of inner loss: MultipleNegativesRankingLoss
    -   wrap inner loss in overall loss: MatryoshkaLoss
5.  Set up finetuning pipeline
    -   This includes data, model, loss and hyperparameters
    -   Hyperparameters include number of epochs, warmup, etc.
6.  Run the finetuning pipeline and get modified model embeddings
    -   save these embeddings
    -   see if these can be loaded onto HF
    -   see if these can be downloaded from HF
7.  Validation Loss
    -   run assessment on val sub-group


"""

# imports
from operator import itemgetter
import pandas as pd
from typing import List
import uuid
import random
import tqdm
import re
import json
import pandas as pd

from torch.utils.data import DataLoader

from sentence_transformers import SentenceTransformer
from sentence_transformers import InputExample
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
from sentence_transformers.evaluation import InformationRetrievalEvaluator

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document


class GenerateQuestionsForContexts:
    def __init__(self, 
                 qa_chat_model_name="gpt-4o-mini",
                 n_questions=3):

        self.qa_chat_model_name = qa_chat_model_name
        # regex pattern used to extract questions from LLM response
        # first group is question number - an integer - followed by a period
        # second group is any character that follows this
        self.regex_pattern = r'(^\d+).(.+)'
        self.n_questions = n_questions

        self.set_up_chat_model()
        self.set_up_question_generation_chain()
        return
    
    def get_unique_id(self, id_set):
        """
        Generate unique id not present in input set of ids
        Input
            a set of unique identifiers
        Returns
            a new unique id not in input set
            updated input set of ids incl the newly generated id
        """
        id = str(uuid.uuid4())
        while id in id_set:
            id = str(uuid.uuid4())
        id_set.add(id)
        return id, id_set

    def set_up_chat_model(self):
        self.qa_chat_model = ChatOpenAI(
            model=self.qa_chat_model_name,
            temperature=0
        )
        return self

    def set_up_question_generation_chain(self):
        qa_prompt = """\
        Given the following context, you must generate questions based on only the provided context.

        You are to generate {n_questions} questions which should be provided in the following format:

        1. QUESTION #1
        2. QUESTION #2
        ...

        Context:
        {context}
        """
        qa_prompt_template = ChatPromptTemplate.from_template(qa_prompt)
        self.question_generation_chain = qa_prompt_template | self.qa_chat_model
        return self

    def create_questions(self, documents, n_questions):
        questions = {}
        relevant_docs = {}

        q_id_set = set()
        for document in tqdm.tqdm(documents):  # note tqdm.tqdm (NOT just tqdm as in original notebook)
            this_question_set = \
                self.question_generation_chain.invoke(
                    {
                        'context': document.page_content, 
                        'n_questions': n_questions
                    }
                )
            for question in this_question_set.content.split("\n"):
                if len(question) > 0:
                    try:
                        q_id, q_id_set  = self.get_unique_id(q_id_set)
                        matched_pattern = re.search(self.regex_pattern, question)  # regex search for n. <question>
                        if len(matched_pattern.group(2)) > 0:
                            questions[q_id] = matched_pattern.group(2).strip()  # extraction of question string
                            relevant_docs[q_id] = [document.metadata["id"]]
                    except Exception:
                        continue
        return questions, relevant_docs


class PrepareDataForFinetuning(GenerateQuestionsForContexts):
    def __init__(self, 
                 chunk_size=None, chunk_overlap=None, len_function=None,
                 lcdocuments=None, run_optional_text_splitter=False,
                 all_splits=None, train_val_test_size=[10, 5, 5],
                 train_val_test_split_type='random',
                 random_seed=69, qa_chat_model_name="gpt-4o-mini",
                 n_questions=2, batch_size=5):
    
        super().__init__(qa_chat_model_name=qa_chat_model_name,
                         n_questions=n_questions)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.len_function = len_function

        self.lcdocuments = lcdocuments
        self.run_optional_text_splitter = run_optional_text_splitter

        self.all_doc_splits = all_splits

        self.train_val_test_size = train_val_test_size
        self.n_train = self.train_val_test_size[0]
        self.n_val = self.train_val_test_size[1]
        self.n_test = self.train_val_test_size[2]
        self.train_val_test_split_type = train_val_test_split_type

        self.random_seed = random_seed
        self.batch_size = batch_size
        return

    def optional_text_splitter(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap  = self.chunk_overlap,
            length_function = self.len_function
        )
        self.all_doc_splits = text_splitter.split_documents(self.lcdocuments.load())
        return self
    
    def attach_unique_ids_to_docs(self):
        id_set = set()
        for docsplit in self.all_doc_splits:
            id, id_set  = self.get_unique_id(id_set)
            docsplit.metadata["id"] = id
        return self

    def simple_train_val_test_splits(self):       
        self.training_splits = self.all_doc_splits[:self.n_train]
        self.val_splits = self.all_doc_splits[self.n_train:self.n_train+self.n_val]
        self.test_splits = self.all_doc_splits[self.n_train+self.n_val:]
        return self

    def randomized_train_val_test_splits(self):
        # set the same seed to be able to replicate the result of
        # random shuffle below
        random.seed(self.random_seed)

        # randomly orders the elements in the list training_documents
        randomly_ordered_documents = self.all_doc_splits.copy()
        random.shuffle(randomly_ordered_documents)

        # assign slices to training, val and test
        self.training_splits = randomly_ordered_documents[:self.n_train]
        self.val_splits = randomly_ordered_documents[self.n_train: self.n_train+self.n_val]
        self.test_splits = randomly_ordered_documents[self.n_train+self.n_val:]
        return self
    
    def get_all_questions(self):
        self.training_questions, self.training_relevant_contexts = \
            self.create_questions(documents=self.training_splits, n_questions=self.n_questions)
        self.val_questions, self.val_relevant_contexts = \
            self.create_questions(documents=self.val_splits, n_questions=self.n_questions)
        self.test_questions, self.test_relevant_contexts = \
            self.create_questions(documents=self.test_splits, n_questions=self.n_questions)
        return self

    def save_dataset_to_jsonl(self, splits, questions, relevant_contexts, jsonl_filename):
        """
        NOTE: Each `jsonl` file has a single line!  This is a nested JSON structure.
        Primary keys for each file are `questions`, `relevant_contexts` and `corpus`.
        1.  Each `question` element is a json object with a key id for the 
                question and the string corresp to question as the value.
        2.  Each `relevant_contexts` element is a json object with key id 
                corresponding to a question id and value corresponding to a unique id for the context
        3.  Each `corpus` element is a json object with key id 
                corresponding to a unique context id and value being the context string.
        """
        corpus = {item.metadata["id"] : item.page_content for item in splits}
        dataset_dict = {
            "questions" : questions,
            "relevant_contexts" : relevant_contexts,
            "corpus" : corpus
        }
        with open(jsonl_filename, "w") as f:
            json.dump(dataset_dict, f)
        return dataset_dict

    def save_train_val_test_dataset_to_jsonl(self):
        self.train_dataset = \
            self.save_dataset_to_jsonl(self.training_splits, 
                                       self.training_questions, 
                                       self.training_relevant_contexts,
                                       jsonl_filename='./data/finetuning_data/training_dataset.jsonl')
        
        self.val_dataset = \
            self.save_dataset_to_jsonl(self.val_splits, 
                                       self.val_questions, 
                                       self.val_relevant_contexts,
                                       jsonl_filename='./data/finetuning_data/val_dataset.jsonl')
        
        self.test_dataset = \
            self.save_dataset_to_jsonl(self.test_splits, 
                                       self.test_questions, 
                                       self.test_relevant_contexts,
                                       jsonl_filename='./data/finetuning_data/test_dataset.jsonl')
        return self
    
    def run_all_prep_data(self):
        # if docs are passed in pre-chunking, then split docs
        if self.run_optional_text_splitter is True:
            self.optional_text_splitter()

        # each chunk i.e., context gets a unique id
        self.attach_unique_ids_to_docs()

        # split into train, val and test - either random or simple slicing
        if self.train_val_test_split_type.upper() == 'RANDOM':
            self.randomized_train_val_test_splits()
        else:
            self.simple_train_val_test_splits()

        # generate questions for each context
        # this step involves large number of LLM calls
        self.get_all_questions()

        # save train, val and test datasets in jsonl format
        self.save_train_val_test_dataset_to_jsonl()
        return self


class FineTuneModel:
    def __init__(self,
                 train_data,
                 val_data,
                 batch_size,
                 base_model_id='Snowflake/snowflake-arctic-embed-m',
                 matryoshka_dimensions=[768, 512, 256, 128, 64],
                 number_of_training_epochs=5,
                 finetuned_model_output_path='finetuned_arctic',
                 evaluation_steps = 50):
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size

        self.base_model_id = base_model_id
        self.matryoshka_dimensions = matryoshka_dimensions
        self.number_of_training_epochs = number_of_training_epochs
        self.finetuned_model_output_path = finetuned_model_output_path
        self.evaluation_steps = evaluation_steps

        self.model = SentenceTransformer(self.base_model_id)
        return
    
    def prepare_data_for_finetuning(self, data):
        corpus = data['corpus']
        queries = data['questions']
        relevant_docs = data['relevant_contexts']
        return corpus, queries, relevant_docs
    
    def get_data_loader(self):
        corpus, queries, relevant_docs = self.prepare_data_for_finetuning(self.train_data)

        examples = []
        for query_id, query in queries.items():
            doc_id = relevant_docs[query_id][0]
            text = corpus[doc_id]
            example = InputExample(texts=[query, text])
            examples.append(example)
        self.loader = DataLoader(examples, batch_size=self.batch_size)
        return self
    
    def loss_function(self):
        inner_training_loss = MultipleNegativesRankingLoss(self.model)
        self.train_loss = MatryoshkaLoss(
            self.model,
            inner_training_loss,
            matryoshka_dims=self.matryoshka_dimensions
        )
        return self
    
    def get_evaluator_for_val(self):
        corpus, queries, relevant_docs = self.prepare_data_for_finetuning(self.val_data)
        self.evaluator = InformationRetrievalEvaluator(queries, corpus, relevant_docs)
        return self
    
    def fit_model(self):
        warmup_steps = int(len(self.loader) * self.number_of_training_epochs * 0.1)
        self.model.fit(
            train_objectives=[(self.loader, self.train_loss)],
            epochs=self.number_of_training_epochs,
            warmup_steps=warmup_steps,
            output_path=self.finetuned_model_output_path,
            show_progress_bar=True,
            evaluator=self.evaluator,
            evaluation_steps=self.evaluation_steps,
        )
    
    def run_steps_to_finetune_model(self):
        # load train data into Loader
        self.get_data_loader()

        # set up loss function
        self.loss_function()
        
        # set up evaluator with val data
        self.get_evaluator_for_val()

        # finetune the model
        self.fit_model()
        return self


class FineTuneModelAndEvaluateRetriever(FineTuneModel):
    def __init__(self,
                 train_data,
                 val_data,
                 test_data,
                 batch_size,
                 base_model_id='Snowflake/snowflake-arctic-embed-m',
                 matryoshka_dimensions=[768, 512, 256, 128, 64],
                 number_of_training_epochs=5,
                 finetuned_model_output_path='finetuned_arctic',
                 evaluation_steps = 50,
                 ):
        super().__init__(train_data=train_data,
                         val_data=val_data,
                         batch_size=batch_size,
                         base_model_id=base_model_id,
                         matryoshka_dimensions=matryoshka_dimensions,
                         number_of_training_epochs=number_of_training_epochs,
                         finetuned_model_output_path=finetuned_model_output_path,
                         evaluation_steps = evaluation_steps)
        self.test_data = test_data
        return
    
    def set_up_test_data_for_retrieval(self, embedding_model_for_retrieval, top_k_for_retrieval):
        corpus, questions, relevant_docs = self.prepare_data_for_finetuning(self.test_data)

        documents = [Document(page_content=content, metadata={"id": doc_id}) 
                     for doc_id, content in corpus.items()]
        
        vectorstore = FAISS.from_documents(documents, embedding_model_for_retrieval)
        retriever = vectorstore.as_retriever(search_kwargs={"k": top_k_for_retrieval})
        return corpus, questions, relevant_docs, retriever
    
    def evaluate_embeddings_model(self, embedding_model_for_retrieval, top_k_for_retrieval, verbose=False):
        corpus, questions, relevant_docs, retriever = \
            self.set_up_test_data_for_retrieval(embedding_model_for_retrieval, top_k_for_retrieval)
        eval_results = []
        for id, question in tqdm.tqdm(questions.items()):
            retrieved_nodes = retriever.invoke(question)
            retrieved_ids = [node.metadata["id"] for node in retrieved_nodes]
            expected_id = relevant_docs[id][0]
            is_hit = expected_id in retrieved_ids
            eval_results.append({"id": id, "question": question, "expected_id": expected_id, "is_hit": is_hit})
        return eval_results
