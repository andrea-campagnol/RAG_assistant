# RAG

Retrieval-Augmented Generation is an approach of Natural Language Processing (NLP) which combines retrieval systems and generative models to generate an accurate and contextually relevant response. It is a two-step process combining external knowledge search with prompting. The retrieval system is designed to seach through large collection of documents (corpus) to find the most relevant pieces of information based on given query or question. In generation model, the fetched document are wrapped into a prompt and passed to LLM (in this case, we are using Meta-Llama-3.1-8B-Instruct) to generate relevant response.

## Objective of the project

The objective of the project is specifically to implement a RAG application on the AI Stack Exchange dataset and evaluate its question-answering capability in the given dataset domain.
