import json
import os

from PyPDF2 import PdfReader
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuração do modelo local
LOCAL_MODEL_DIR = (
    "./local_openllama_model"  # Caminho onde o modelo está armazenado localmente
)
CHUNK_SIZE = 500  # Limite de tokens por chunk


# Função para carregar o modelo localmente
def load_local_model(local_path):
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Modelo não encontrado no caminho: {local_path}")
    print(f"Carregando modelo local de: {local_path}")

    tokenizer = AutoTokenizer.from_pretrained(local_path)
    model = AutoModelForCausalLM.from_pretrained(
        local_path, device_map="cpu"
    )  # Força uso de CPU
    return model, tokenizer


# Carregar o modelo e o tokenizer
model, tokenizer = load_local_model(LOCAL_MODEL_DIR)


# Função para extrair texto dos PDFs e dividir em chunks
def read_pdf_in_chunks(pdf_path, chunk_size=CHUNK_SIZE):
    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    # Dividir o texto em pedaços limitados por tokens
    tokens = tokenizer.encode(text)
    chunks = [tokens[i : i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    return [tokenizer.decode(chunk) for chunk in chunks]


# Função para gerar perguntas e respostas para o texto completo
def generate_questions_answers_from_text(chunks):
    full_content = []

    for context in chunks:
        # Gerar pergunta com base no contexto
        prompt_question = f"""
Below is a context from a document. Write a question based on the context and provide a concise answer.

### Context:
{context}

### Question:
"""
        inputs = tokenizer(prompt_question, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=50)
        question = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Gerar resposta com base na pergunta
        prompt_answer = prompt_question + question + "\n\n### Answer:"
        inputs = tokenizer(prompt_answer, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=50)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Adicionar o contexto, pergunta e resposta
        full_content.append(
            {
                "context": context,
                "question": question,
                "answer": answer,
            }
        )

    return full_content


# Criar dataset com perguntas e respostas para cada PDF
def create_dataset_from_pdfs(folder_path, output_path, chunk_size=CHUNK_SIZE):
    dataset = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(folder_path, file_name)
            print(f"Processando: {file_name}")

            # Extrair texto do PDF em chunks
            chunks = read_pdf_in_chunks(file_path, chunk_size=chunk_size)

            # Gerar perguntas e respostas para todo o conteúdo
            qa_content = generate_questions_answers_from_text(chunks)

            # Adicionar cada entrada ao dataset
            for entry in qa_content:
                dataset.append(
                    {
                        "file_name": file_name,
                        "context": entry["context"],
                        "question": entry["question"],
                        "answer": entry["answer"],
                    }
                )

    # Salvar dataset como JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

    print(f"Dataset salvo em: {output_path}")


# Configuração da pasta e execução
folder_path = "pdfs"  # Caminho para os PDFs
output_path = "pdf_full_question_answer_dataset.json"  # Caminho de saída do dataset
create_dataset_from_pdfs(folder_path, output_path)
