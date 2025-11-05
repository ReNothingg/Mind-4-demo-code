import os
import sys
import time
import traceback
from pathlib import Path

import PyPDF2
from nltk.tokenize import TreebankWordTokenizer

def extract_text_from_pdf(pdf_path):
    text = ""
    reader = PyPDF2.PdfReader(str(pdf_path), strict=False)
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def tokenize_and_save(text, output_path):
    tokenizer = TreebankWordTokenizer()
    tokens = tokenizer.tokenize(text)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for token in tokens:
            f.write(token + '\n')
    return len(tokens)

def find_pdfs(input_dir):
    return list(Path(input_dir).rglob("*.pdf"))

def main():
    base_input = Path("./Dataset")
    base_output = Path("./Tokenizer")

    pdf_files = find_pdfs(base_input)
    total_files = len(pdf_files)
    if total_files == 0:
        print(f"Ни одного PDF в папке {base_input.resolve()}")
        sys.exit(1)

    print(f"Найдено PDF-файлов: {total_files}\n")
    overall_tokens = 0
    error_count = 0
    start_time = time.time()

    for idx, pdf_path in enumerate(pdf_files, start=1):
        rel_path = pdf_path.relative_to(base_input)
        out_file = base_output / rel_path.with_suffix(".tokens.txt")

        print(f"[{idx}/{total_files}] 🔄 Обрабатываем: {pdf_path}")
        file_start = time.time()

        try:
            text = extract_text_from_pdf(pdf_path)
            num_tokens = tokenize_and_save(text, out_file)
            overall_tokens += num_tokens
            elapsed = time.time() - file_start
            percent = idx / total_files * 100
            print(f"Сохранено {num_tokens} токенов -> {out_file}")
            print(f"Время: {elapsed:.1f}s | Прогресс: {percent:.1f}%\n")
        except Exception as e:
            error_count += 1
            elapsed = time.time() - file_start
            print(f"Ошибка при обработке: {e.__class__.__name__}")
            print(f"{pdf_path} пропущен (время попытки: {elapsed:.1f}s)\n")
            continue

    total_time = time.time() - start_time
    print("="*40)
    print(f"Готово! Всего файлов: {total_files}")
    print(f"Успешно: {total_files - error_count}")
    print(f"Пропущено: {error_count}")
    print(f"Всего токенов: {overall_tokens}")
    print(f"Общее время: {total_time:.1f}s")
    print("="*40)

if __name__ == "__main__":
    main()
