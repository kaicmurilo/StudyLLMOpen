import csv
import os

from PIL import Image
from transformers import AutoProcessor, Pix2StructForConditionalGeneration

# Nome do modelo e diretório local
model_name = "google/deplot"
local_path = "./models/deplot"
image_dir = "./imagens/no_extracted"
output_dir = "./csv/extracted_images"

# Verifica se o modelo já foi baixado
if not os.path.exists(local_path):
    print("Modelo não encontrado localmente. Baixando...")
    processor = AutoProcessor.from_pretrained(model_name)
    processor.save_pretrained(local_path)

    model = Pix2StructForConditionalGeneration.from_pretrained(model_name)
    model.save_pretrained(local_path)
else:
    print("Modelo encontrado localmente. Carregando...")
    processor = AutoProcessor.from_pretrained(local_path)
    model = Pix2StructForConditionalGeneration.from_pretrained(local_path)

# Verifica e cria a pasta de saída, caso não exista
os.makedirs(output_dir, exist_ok=True)

# Processa cada imagem na pasta de entrada
for image_file in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_file)
    if not image_file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
        print(f"Arquivo ignorado (não é uma imagem): {image_file}")
        continue

    print(f"Processando imagem: {image_file}")
    image = Image.open(image_path)

    # Processa a imagem
    inputs = processor(
        images=image,
        text="Generate underlying data table of the figure below:",
        return_tensors="pt",
        legacy=False,
    )

    # Gera os dados do gráfico
    predictions = model.generate(**inputs, max_new_tokens=512)

    # Decodifica os resultados
    output = processor.decode(predictions[0], skip_special_tokens=True)

    # Limpa os caracteres especiais e formata os dados
    cleaned_output = output.replace("<0x0A>", "\n").strip()
    data_lines = cleaned_output.split("\n")  # Divide em linhas

    # Verifica se há cabeçalhos e dados
    if len(data_lines) < 2:
        print(f"Dados insuficientes para {image_file}, ignorando...")
        continue

    headers = data_lines[0].split("|")  # Assume que o cabeçalho usa "|"
    rows = [line.split("|") for line in data_lines[1:]]

    # Salva os dados no CSV
    csv_file_name = f"{os.path.splitext(image_file)[0]}.csv"
    csv_path = os.path.join(output_dir, csv_file_name)
    with open(csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([h.strip() for h in headers])  # Escreve os cabeçalhos
        for row in rows:
            csv_writer.writerow([r.strip() for r in row])  # Escreve as linhas

    print(f"Dados salvos em: {csv_path}")
