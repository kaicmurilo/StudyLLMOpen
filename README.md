# Study of the Use of LLMs to Answer Questions About Training Topics

This project explores the use of large language models (LLMs) to answer questions based on specific training materials.

## Current Status

The application is in progress. I am working on fine-tuning the model using Google Colab, but the implementation is not finalized at the moment.
The code file starting with '\_' is functional. Other files are not functional, but they are in progress.

## Steps to Run

### 1.1 Download the Model in manual mode

Use the Hugging Face CLI to download the model:

```bash
huggingface-cli download openlm-research/open_llama_3b_v2
```

### 1.2 Download the Model in automatic mode

Using in moment llama, but yout can use other models.

```bash
python _download_llama3-2-1-b.py
```

### 2. Create a Dataset from PDFs (not functional)

The current dataset creation process extracts text from PDFs and structures it into a format compatible with the model. The dataset includes context and questions while respecting the token limit of the model.

To generate datasets from your PDF files, use the following script:

```bash
python create_datasets.py
```

### 3. Create Vetorial Search (Functional)

Read PDFs and create a vectorial search. Reads the PDF document and vectorizes it by sentences, without performing OCR. Validates if the PDF is a text-based document.

```bash
python _create_vetorial.py
```

### 4. Run the Application (functional with vetorial search)

Run the application with Gradio:

```bash
python _app_interface_agent.py
```

## Goal

The primary goal of this project is to train a model on custom documents and provide an API to answer questions based on the content of those documents.
