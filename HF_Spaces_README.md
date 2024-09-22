---
title: AIE4 Midterm - AI Risks ChatBot
emoji: 🌖
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: apache-2.0
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference


# Summary

This is my app for AI Engineering Cohort#4 Midterm Assignment.

With this application, you can chat with these TWO uploaded PDFs:
https://www.whitehouse.gov/wp-content/uploads/2022/10/Blueprint-for-an-AI-Bill-of-Rights.pdf
AND
https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf


Code features:
1.  Text splitter - RecursiveTextSplitter from Langchain
2.  Vector store - using Qdrant db
3.  Retrieval chain using LCEL syntax
4.  Chat model is OpenAI's gpt-4o-mini.
5.  There are two variants of the app that can be deployed:
    a.  An early prototype built using OpenAI Embeddings text-embedding-3-small
    b.  A more advanced prototype using finetuned version of `Snowflake/snowflake-arctic-embed-m`
        the finetuned version is available at `vincha77/finetuned_arctic`
