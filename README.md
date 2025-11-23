# Hybrid-RAG-Fine-Tuned-Clinical-Assistant-With-Vertex-AI

Prerequisites
Before running the code, ensure you have:

Google Cloud Platform (GCP) Account: A project with billing enabled.

Vertex AI API Enabled: Go to the GCP Console > Vertex AI > Enable API.

Python 3.9+ installed.

Google Cloud SDK (gcloud) installed and authenticated.

MedQuAD Assistant: RAG & Fine-Tuning with Google Vertex AI
This project demonstrates a hybrid AI approach to building a specialized Medical Question Answering system. By leveraging the MedQuAD dataset, this solution combines Retrieval-Augmented Generation (RAG) for factual accuracy with Supervised Fine-Tuning for domain-specific behavioral adaptation using Google's Gemini models via Vertex AI.

##Project Overview
In the medical domain, accuracy and tone are critical. A standard LLM might hallucinate facts or use casual language. This project solves that by implementing a two-tiered architecture:

RAG (Retrieval-Augmented Generation): Ensures the model answers questions based strictly on the provided medical dataset (medquad.csv), minimizing hallucinations.

Fine-Tuning (Vertex AI): Adapts the generic Gemini model to better understand medical terminology and structure its responses in a professional, clinical format.

##Architecture

The pipeline is built on Google Cloud Platform (Vertex AI):

Data Source: [MedQuAD (Medical Question Answering Dataset)](https://www.kaggle.com/datasets/jpmiller/layoutlm).

Embeddings: text-embedding-004 for high-quality semantic vectorization.

LLM: gemini-1.5-flash-002 (chosen for its speed and reasoning capabilities).

Vector Search: In-memory cosine similarity (scalable to Vertex AI Vector Search for production).

Why RAG + Fine-Tuning?
While RAG provides the "knowledge" (the specific facts), Fine-Tuning provides the "form" (the style and expertise).

Fine-Tuning teaches the model: "Act like a doctor."

RAG gives the model: "Here is the specific diagnosis data for this patient." Using both ensures high-quality, context-aware, and stylistically correct responses.

How to Run
Step 1: Data Preparation & Fine-Tuning
Run this script to convert the CSV into JSONL format required by Vertex AI and (optionally) submit a fine-tuning job.

Bash

python data_prep_tuning.py

Note: To actually trigger the training job in the cloud, ensure you uncomment the execution lines in the script and provide a valid Google Cloud Storage bucket URI.

Step 2: Launch the RAG System
Run the main application to interact with the medical assistant. This script indexes the data locally and opens a chat interface in your terminal.

Bash

python rag_system.py

Example Interaction:

Enter a medical question: What are the symptoms of Diabetes?

[System]: Retrieving context... [Model Response]: Based on the clinical data, the symptoms include increased thirst, frequent urination, hunger, fatigue, and blurred vision...

## Author
Steven Deaquis Systems Engineer | Certified Data Scientist | Bilingual ML Developer Focused on reproducible pipelines, international deployment, and performance tracking.
