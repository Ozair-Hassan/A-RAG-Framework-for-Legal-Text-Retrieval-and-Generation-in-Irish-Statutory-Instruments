# A RAG Framework for Legal Text Retrieval and Generation in Irish Statutory Instruments

This repository contains the code and experiments for the MSc thesis:

**“A RAG Framework for Legal Text Retrieval and Generation in Irish Statutory Instruments.”**

## 1. Requirements

### Python

- **Python 3.10+**

### Install dependencies

```bash

pip install requests beautifulsoup4 tqdm unidecode pdfminer.six pytesseract pdfplumber Pillow nltk datasets pandas scikit-learn rank-bm25 sentence-transformers contractions groq python-dotenv nest_asyncio weaviate-client matplotlib backoff openai

```

Optional:

- trec_eval
- Docker
- Tesseract OCR

---

## 2. Installation

```bash

git clone https://github.com/Ozair-Hassan/A-RAG-Framework-for-Legal-Text-Retrieval-and-Generation-in-Irish-Statutory-Instruments
cd A-RAG-Framework-for-Legal-Text-Retrieval-and-Generation-in-Irish-Statutory-Instruments

```

Create environment:

Powershell:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Gitbash:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. End-to-End Pipeline

### Step 1 – Scrape PDFs

```bash
python scrape.py --download-path ./downloads --links-file Links.txt --delay 1.0
```

### Step 2 – Preprocess PDFs

```bash
python preprocess.py
```

### Step 3 – Setup env

Create `.env` in the same directory level as the jupyter notebooks:

```
GROQ_API_KEY_1=your_key
```

### Step 4 – Baseline

Run:

- `Benchmark/DP_Au_Irish.ipynb`
- `Benchmark/Eval_Au_Irish.ipynb`
- `Benchmark/QnA_Au_Irish.ipynb`

### Step 4 – Weaviate setup

Install docker desktop from https://www.docker.com/products/docker-desktop/

Run:

```bash
cd weaviate_local
docker compose up -d
```

Once docker container is up and status is 200

```bash
python view.py
```

To test Weaviate and ensure it is working as needed

### Step 5 – Weaviate RAG

Run:

- `Weaviate_Implementation/DP_My_Irish.ipynb`
- `Weaviate_Implementation/Eval_My_Irish.ipynb`
- `Weaviate_Implementation/QnA_My_Irish.ipynb`

### Step 6 – RePASs Evaluation

Clone RePASs repo to validate the results

```bash
git clone https://github.com/RegNLP/RePASs.git && cd RePASs
```

Next run the following

```bash
python scripts/evaluate_model.py --input_file answers.json --group_method_name hybrid-llama
```

---

## Contact

Open an issue on the repository.

## MIT License

Copyright (c) 2025 Ozair Hassan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

test
