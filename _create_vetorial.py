import os
import pickle

import faiss
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer


# Passo 1: Extrair texto do PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()  # Extrai texto de cada página
    return text


# Passo 2: Dividir texto em frases
def split_into_sentences(text):
    import re

    # Divide com base em pontuação
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


# Passo 3: Gerar embeddings com SentenceTransformers
def generate_embeddings(sentences, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences, convert_to_numpy=True)
    return embeddings


# Passo 4: Criar ou atualizar banco vetorial com FAISS
def create_or_update_faiss_index(index, embeddings, sentences, id_to_sentence):
    if index is None:  # Criar novo índice
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)  # Adiciona os vetores ao índice

    # Atualizar o mapeamento de sentenças
    current_offset = len(id_to_sentence)
    for i, sentence in enumerate(sentences):
        id_to_sentence[current_offset + i] = sentence

    return index, id_to_sentence


# Passo 5: Salvar banco vetorial e mapeamento
def save_faiss_index(index, id_to_sentence, index_path, mapping_path):
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
    with open(mapping_path, "wb") as f:
        pickle.dump(id_to_sentence, f)


# Passo 6: Carregar banco vetorial e mapeamento
def load_faiss_index(index_path, mapping_path):
    index = faiss.read_index(index_path)
    with open(mapping_path, "rb") as f:
        id_to_sentence = pickle.load(f)
    return index, id_to_sentence


# Processar todos os PDFs de um diretório
def process_pdfs(
    pdf_dir, trained_dir, index_path, mapping_path, model_name="all-MiniLM-L6-v2"
):
    # Carregar banco vetorial existente ou inicializar novo
    if os.path.exists(index_path) and os.path.exists(mapping_path):
        print("Carregando banco vetorial existente...")
        index, id_to_sentence = load_faiss_index(index_path, mapping_path)
    else:
        print("Criando novo banco vetorial...")
        index, id_to_sentence = None, {}

    # Processar PDFs
    for pdf_file in os.listdir(pdf_dir):
        pdf_path = os.path.join(pdf_dir, pdf_file)
        if pdf_file.endswith(".pdf"):
            print(f"Processando {pdf_file}...")
            try:
                # Extrair texto e gerar embeddings
                text = extract_text_from_pdf(pdf_path)
                sentences = split_into_sentences(text)
                embeddings = generate_embeddings(sentences, model_name)

                # Atualizar o banco vetorial
                index, id_to_sentence = create_or_update_faiss_index(
                    index, embeddings, sentences, id_to_sentence
                )

                # Salvar banco vetorial atualizado
                save_faiss_index(index, id_to_sentence, index_path, mapping_path)
                print(f"Banco vetorial atualizado com sucesso para {pdf_file}.")

                # Mover arquivo para a pasta de treinados
                os.makedirs(trained_dir, exist_ok=True)
                os.rename(pdf_path, os.path.join(trained_dir, pdf_file))
            except Exception as e:
                print(f"Erro ao processar {pdf_file}")
                raise e

    # Salvar banco vetorial atualizado
    save_faiss_index(index, id_to_sentence, index_path, mapping_path)
    print("Banco vetorial atualizado e salvo.")
    return index, id_to_sentence


# Exemplo de execução
if __name__ == "__main__":
    # Diretórios de entrada e saída
    pdf_dir = "pdfs/no_train/"
    trained_dir = "pdfs/trained/"

    # Caminhos para o banco vetorial
    index_path = "map_local/faiss_index.bin"
    mapping_path = "map_local/id_to_sentence.pkl"

    # Processar PDFs e atualizar o banco vetorial
    index, id_to_sentence = process_pdfs(pdf_dir, trained_dir, index_path, mapping_path)
