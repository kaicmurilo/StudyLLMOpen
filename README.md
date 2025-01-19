# Study of the Use of LLMs to Answer Questions About Training Topics

This project explores the use of large language models (LLMs) to answer questions based on specific training materials.

## Current Status
The application is in progress. I am working on fine-tuning the model using Google Colab, but the implementation is not finalized at the moment.

## Steps to Run

### 1. Download the Model
Use the Hugging Face CLI to download the model:

```bash
huggingface-cli download openlm-research/open_llama_3b_v2
```

### 2. Create a Dataset from PDFs

The current dataset creation process extracts text from PDFs and structures it into a format compatible with the model. The dataset includes context and questions while respecting the token limit of the model. 

To generate datasets from your PDF files, use the following script:


```bash
python create_datasets.py
```

### 3. Run the Application
Run the application with Gradio:

```bash
python app_interface_agent.py
```


## Goal
The primary goal of this project is to train a model on custom documents and provide an API to answer questions based on the content of those documents.
