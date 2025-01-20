import hashlib
import json
import os

from PyPDF2 import PdfReader
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuração do modelo e do chunk
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"  # Nome do modelo a ser usado
LOCAL_MODEL_DIR = "./models/Llama-3.2-1B-Instruct"  # Caminho do modelo local
CHUNK_SIZE = 500  # Limite de tokens por chunk
CONFIG_DIR = "config"  # Diretório de configuração
HASH_FILE = os.path.join(
    CONFIG_DIR, "processed_files.json"
)  # Hash dos arquivos processados


# Função para calcular o hash de um arquivo
def calculate_file_hash(file_path):
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


# Função para carregar o modelo localmente
def load_local_model(local_path):
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Modelo não encontrado no caminho: {local_path}")
    print(f"Carregando modelo local de: {local_path}")

    tokenizer = AutoTokenizer.from_pretrained(local_path)
    model = AutoModelForCausalLM.from_pretrained(
        local_path, device_map="auto"  # Ajusta automaticamente o dispositivo (CPU/GPU)
    )
    return model, tokenizer


# Função para carregar hashes dos arquivos processados
def load_processed_hashes():
    if not os.path.exists(HASH_FILE):
        os.makedirs(CONFIG_DIR, exist_ok=True)
        with open(HASH_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)
    with open(HASH_FILE, "r", encoding="utf-8") as f:
        return set(json.load(f))


# Função para salvar hashes dos arquivos processados
def save_processed_hash(file_hash):
    hashes = load_processed_hashes()
    hashes.add(file_hash)
    with open(HASH_FILE, "w", encoding="utf-8") as f:
        json.dump(list(hashes), f, ensure_ascii=False, indent=4)


# Função para extrair texto dos PDFs e dividir em chunks
def read_pdf_in_chunks(pdf_path, chunk_size=CHUNK_SIZE):
    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    # Dividir o texto em pedaços limitados por tokens
    tokens = tokenizer.encode(text)
    chunks = [tokens[i : i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]


# Função para gerar perguntas e respostas para o texto completo
def generate_questions_answers_from_text(chunks):
    full_content = []

    for context in chunks:
        # Gerar pergunta com base no contexto
        prompt_question = f"""
            Abaixo está um contexto de um documento. Escreva uma pergunta com base no contexto e forneça uma resposta concisa.

            ### Contexto:
            {context}

            ### Pergunta:
            """
        inputs = tokenizer(prompt_question, return_tensors="pt", truncation=True)
        outputs = model.generate(inputs["input_ids"], max_new_tokens=50)
        question = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Gerar resposta com base na pergunta
        prompt_answer = prompt_question + question + "\n\n### Resposta:"
        inputs = tokenizer(prompt_answer, return_tensors="pt", truncation=True)
        outputs = model.generate(inputs["input_ids"], max_new_tokens=50)
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
    processed_hashes = load_processed_hashes()

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(folder_path, file_name)
            file_hash = calculate_file_hash(file_path)

            # Ignorar arquivos já processados
            if file_hash in processed_hashes:
                print(f"Ignorando {file_name}, já processado.")
                continue

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

            # Salvar hash do arquivo processado
            save_processed_hash(file_hash)

    # Salvar dataset como JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

    print(f"Dataset salvo em: {output_path}")


# Configuração da pasta e execução
folder_path = "pdfs"  # Caminho para os PDFs
output_path = (
    "map_local/pdf_full_question_answer_dataset.json"  # Caminho de saída do dataset
)
model, tokenizer = load_local_model(LOCAL_MODEL_DIR)
create_dataset_from_pdfs(folder_path, output_path)
