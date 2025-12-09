import os
import re
import sys
import json
import hashlib
from pathlib import Path
import traceback
from typing import List, Dict, Optional, Tuple

from tqdm import tqdm
from unidecode import unidecode # type: ignore
from pdfminer.high_level import extract_text as pdfminer_extract_text # type: ignore

INPUT_DIR = "./downloads"                   
OUTPUT_DIR = "./output"                    
GLOBAL_JSONL = None                         
ENABLE_OCR = False                          
YEARS_TO_PROCESS = [2025, 2024, 2023, 2022, 2021, 2020]

# OCR dependencies
try:
    import pytesseract # type: ignore
    from PIL import Image
    import pdfplumber # type: ignore
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# NLTK for text normalization
try:
    import nltk # type: ignore
    from nltk.corpus import stopwords # type: ignore
    from nltk.stem import WordNetLemmatizer # type: ignore
except ImportError:
    print("ERROR: NLTK not installed. Run: pip install nltk")
    sys.exit(1)



class Config:
    # PDF extraction
    MIN_TEXT_LENGTH = 50
    OCR_RESOLUTION = 300
    
    # Text normalization
    KEEP_LEGAL_TERMS = ["shall", "may", "must"]
    
    # Regex patterns
    SI_PATTERN = re.compile(r"S\.I\.\s*No\.\s*(\d+)\s*of\s*(\d{4})", re.IGNORECASE)
    TITLE_KEYWORDS = r"(ORDER|REGULATION|REGULATIONS|ACT|COMMENCEMENT|APPOINTMENT|STATISTICS|DELEGATION|SCHEDULE)"
    
    HEADER_PATTERNS = {
        "explicit_reg": re.compile(r"^(?:Regulation|REGULATION|Reg\.)\s+(\d+)\.", re.MULTILINE),
        "bare_number": re.compile(r"^(\d+)\.\s?\—?|^(\d+)\.\s", re.MULTILINE),
        "schedule": re.compile(r"^(SCHEDULE(?:\s+\d+)?)\b", re.MULTILINE),
        "explanatory": re.compile(r"^(EXPLANATORY\s+NOTE)\b", re.MULTILINE),
    }


def ensure_nltk_data():
    print("Checking NLTK data...")
    for resource in ["corpora/stopwords", "corpora/wordnet"]:
        try:
            nltk.data.find(resource)
            print(f"Found {resource}")
        except LookupError:
            package = resource.split("/")[1]
            print(f"Downloading {package}...")
            nltk.download(package, quiet=True)
            print(f"Downloaded {package}")


def initialize_text_tools() -> Tuple[set, WordNetLemmatizer]:
    try:
        ensure_nltk_data()
        
        stop_words = set(stopwords.words("english"))
        for term in Config.KEEP_LEGAL_TERMS:
            stop_words.discard(term)
        
        lemmatizer = WordNetLemmatizer()
        print(f"Initialized text tools (stopwords: {len(stop_words)}, legal terms kept: {len(Config.KEEP_LEGAL_TERMS)})")
        return stop_words, lemmatizer
    except Exception as e:
        print(f"Failed to initialize text tools: {e}")
        traceback.print_exc()
        sys.exit(1)


# PDF Extraction

def extract_text_pdfminer(pdf_path: Path) -> str:
    try:
        text = pdfminer_extract_text(str(pdf_path)) or ""
        if text:
            print(f" extracted {len(text)} characters from {pdf_path.name}")
        return text
    except Exception as e:
        print(f" failed for {pdf_path.name}: {e}")
        return ""


def extract_text_ocr(pdf_path: Path) -> str:
    if not OCR_AVAILABLE:
        print(f"  OCR not available")
        return ""
    
    print(f" Running OCR on {pdf_path.name}...")
    text_chunks = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                img = page.to_image(resolution=Config.OCR_RESOLUTION)
                pil_img = Image.frombytes(
                    "RGB",
                    img.original.size,
                    img.original.tobytes()
                )
                text = pytesseract.image_to_string(pil_img)
                if text:
                    text_chunks.append(text)
                print(f" Page {i}/{len(pdf.pages)} processed")
        
        full_text = "\n".join(text_chunks)
        print(f" OCR extracted {len(full_text)} characters")
        return full_text
    except Exception as e:
        print(f" OCR failed for {pdf_path.name}: {e}")
        traceback.print_exc()
        return ""


def calculate_checksum(file_path: Path) -> str:
    try:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return f"sha256:{h.hexdigest()}"
    except Exception as e:
        print(f"  Checksum calculation failed: {e}")
        return "sha256:error"


# Text Cleaning 

def clean_whitespace(text: str) -> str:
    original_len = len(text)
    text = text.replace("\r", "")
    text = re.sub(r"-\n(?=\w)", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    cleaned = text.strip()
    print(f" Cleaned text: {original_len} → {len(cleaned)} characters")
    return cleaned


def normalize_for_retrieval(text: str, stop_words: set, lemmatizer: WordNetLemmatizer) -> str:

    try:
        text = unidecode(text).lower()
        text = re.sub(r"[^a-z0-9\-\(\)\.\s]", " ", text)
        text = re.sub(r"\s{2,}", " ", text)
        
        tokens = []
        for token in text.split():
            if token in stop_words:
                continue
            token = token.strip(".")
            tokens.append(lemmatizer.lemmatize(token))
        
        normalized = " ".join(tokens).strip()
        print(f"    Normalized: {len(text.split())} → {len(tokens)} tokens")
        return normalized
    except Exception as e:
        print(f"  Normalization failed: {e}")
        return text.lower()


# Metadata Extraction 

def extract_metadata(text: str) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    si_match = Config.SI_PATTERN.search(text)
    si_number_str = None
    year = None
    
    if si_match:
        num, yr = si_match.groups()
        si_number_str = f"S.I. No. {int(num)} of {int(yr)}"
        year = int(yr)
        print(f" Found S.I.: {si_number_str}")
    else:
        print(f" No S.I. number found in document")
    
    # Extract title
    title = None
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    
    if si_match:
        si_line_idx = 0
        for i, line in enumerate(lines):
            if Config.SI_PATTERN.search(line):
                si_line_idx = i
                break
        
        window = lines[si_line_idx:si_line_idx + 20]
        candidates = []
        for line in window:
            if len(line) >= 8 and re.search(Config.TITLE_KEYWORDS, line, re.IGNORECASE):
                candidates.append(line)
        
        if candidates:
            title = max(candidates, key=len)
            print(f" Extracted title: {title[:60]}...")
    
    if title:
        title = re.sub(r"\s{2,}", " ", title).strip(" _-")
    
    return si_number_str, year, title


def make_doc_id(si_number_str: Optional[str], year: Optional[int], pdf_path: Path) -> str:
    if si_number_str and year:
        match = re.search(r"No\.\s*(\d+)\s*of\s*(\d{4})", si_number_str, re.IGNORECASE)
        if match:
            num = int(match.group(1))
            yr = int(match.group(2))
            doc_id = f"si-{yr}-{num:04d}"
            print(f" Generated doc_id from S.I.: {doc_id}")
            return doc_id
    
    # Fallback: parse filename
    stem = pdf_path.stem.lower()
    year_match = re.search(r"(20\d{2})", stem)
    num_match = re.search(r"(\d{1,4})", stem)
    
    if year_match and num_match:
        doc_id = f"si-{year_match.group(1)}-{int(num_match.group(1)):04d}"
        print(f" Generated doc_id from filename: {doc_id}")
        return doc_id
    
    # Last resort: hash
    doc_id = f"si-{hashlib.sha1(stem.encode()).hexdigest()[:10]}"
    print(f" Generated doc_id from hash: {doc_id}")
    return doc_id


# Section Detection 

def pick_regulation_pattern(text: str) -> str:
    counts = {
        key: len(pattern.findall(text))
        for key, pattern in Config.HEADER_PATTERNS.items()
        if key in ("explicit_reg", "bare_number")
    }
    chosen = max(counts.items(), key=lambda x: x[1])[0]
    print(f"   Detected regulation pattern: {chosen} ({counts[chosen]} matches)")
    return chosen


def find_headers(text: str, reg_pattern_key: str) -> List[Tuple[str, int, int]]:
    headers = []
    
    # Regulation headers
    if reg_pattern_key == "explicit_reg":
        for match in Config.HEADER_PATTERNS["explicit_reg"].finditer(text):
            num = match.group(1)
            headers.append((f"Regulation {num}.", match.start(), match.end()))
    else:  # bare_number
        for match in Config.HEADER_PATTERNS["bare_number"].finditer(text):
            num = match.group(1) or match.group(2)
            if num:
                headers.append((f"Regulation {num}.", match.start(), match.end()))
    
    # Schedule headers
    for match in Config.HEADER_PATTERNS["schedule"].finditer(text):
        label = match.group(1).strip()
        headers.append((label, match.start(), match.end()))
    
    # Explanatory Note
    for match in Config.HEADER_PATTERNS["explanatory"].finditer(text):
        headers.append(("EXPLANATORY NOTE", match.start(), match.end()))
    
    headers.sort(key=lambda x: x[1])
    print(f"  Found {len(headers)} section headers")
    return headers


def slice_sections(text: str, headers: List[Tuple[str, int, int]]) -> List[Dict]:
    sections = []
    for i, (label, start, end) in enumerate(headers):
        next_start = headers[i + 1][1] if i + 1 < len(headers) else len(text)
        full_text = text[start:next_start].strip()
        
        sections.append({
            "section_label": label,
            "start_char": start,
            "end_char": next_start,
            "text_raw": full_text
        })
        print(f"    Section {i+1}: {label} ({len(full_text)} chars)")
    
    return sections


def derive_section_id(label: str) -> str:
    label_norm = label.strip().lower()
    
    if label_norm.startswith("regulation"):
        match = re.search(r"(\d+)", label_norm)
        return f"reg-{int(match.group(1))}" if match else "reg-unknown"
    
    if label_norm.startswith("schedule"):
        match = re.search(r"schedule\s+(\d+)", label_norm)
        return f"schedule-{int(match.group(1))}" if match else "schedule"
    
    if "explanatory note" in label_norm:
        return "explanatory-note"
    
    # Fallback
    return re.sub(r"[^a-z0-9]+", "-", label_norm).strip("-") or "section"


def extract_heading(text_raw: str, section_label: str) -> Optional[str]:
    lines = [l.strip() for l in text_raw.splitlines() if l.strip()]
    if not lines:
        return None
    
    start_idx = 1 if lines[0].lower().startswith(section_label.lower()[:10]) else 0
    
    for i in range(start_idx, min(start_idx + 5, len(lines))):
        if len(lines[i]) > 6:
            return lines[i]
    
    return None


# Save

def write_jsonl(path: Path, records: List[Dict]):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f" Wrote {len(records)} records to {path}")
    except Exception as e:
        print(f" Failed to write {path}: {e}")
        traceback.print_exc()


def append_jsonl(path: Path, records: List[Dict]):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f" Appended {len(records)} records to global file")
    except Exception as e:
        print(f" Failed to append to {path}: {e}")
        traceback.print_exc()


def process_pdf(
    pdf_path: Path,
    out_dir: Path,
    global_jsonl: Optional[Path],
    enable_ocr: bool,
    stop_words: set,
    lemmatizer: WordNetLemmatizer
) -> Tuple[str, int, str]:

    print(f"Processing: {pdf_path.name}")
    
    try:
        # 1) Extract text
        text = extract_text_pdfminer(pdf_path)
        extraction_method = "pdfminer"
        
        if not text or len(text.strip()) < Config.MIN_TEXT_LENGTH:
            print(f"  Insufficient text ({len(text)} chars), trying OCR...")
            if enable_ocr:
                text = extract_text_ocr(pdf_path)
                extraction_method = "ocr" if text else extraction_method
            else:
                print(f"  OCR disabled, skipping")
        
        if not text or len(text.strip()) < Config.MIN_TEXT_LENGTH:
            print(f" FAILED: Insufficient text extracted ({len(text)} chars)")
            return "error", 0, "failed"
        
        text = clean_whitespace(text)
        
        # 2) Extract metadata
        print(f"\n  Extracting metadata...")
        si_number_str, year, title = extract_metadata(text)
        doc_id = make_doc_id(si_number_str, year, pdf_path)
        
        # 3) Detect sections
        print(f"\n  Detecting sections...")
        reg_pattern_key = pick_regulation_pattern(text)
        headers = find_headers(text, reg_pattern_key)
        
        if not headers:
            print(f"  No headers detected, treating as single document")
            headers = [("DOCUMENT", 0, 0)]
        
        sections = slice_sections(text, headers)
        
        # 4) Save
        print(f"\n  Building JSONL records...")
        checksum = calculate_checksum(pdf_path)
        parent_doc_id = doc_id  
        
        records = []
        for sec in sections:
            section_label = sec["section_label"]
            section_id = derive_section_id(section_label)
            text_raw = sec["text_raw"]
            text_norm = normalize_for_retrieval(text_raw, stop_words, lemmatizer)
            heading = extract_heading(text_raw, section_label)
            
            unique_doc_id = f"{parent_doc_id}-{section_id}"
            
            record = {
                "doc_id": unique_doc_id,
                "parent_doc_id": parent_doc_id,
                "section_id": section_id,
                "section_label": section_label,
                "si_number": si_number_str,
                "title": title,
                "year": year,
                "heading": heading,
                "text_raw": text_raw,
                "text_norm": text_norm,
                "start_char": sec["start_char"],
                "end_char": sec["end_char"],
                "source_path": str(pdf_path),
                "extraction_method": extraction_method,
                "checksum": checksum,
                "cross_refs": []
            }
            records.append(record)
        
        print(f"  Created {len(records)} section records")
        
        print(f"\n  Writing output files...")
        output_filename = pdf_path.stem + ".jsonl"
        out_file = out_dir / output_filename
        write_jsonl(out_file, records)
        
        if global_jsonl:
            append_jsonl(global_jsonl, records)
        
        print(f"\n SUCCESS: {output_filename} ({len(records)} sections)")
        return doc_id, len(records), extraction_method
        
    except Exception as e:
        print(f"\n  ERROR processing {pdf_path.name}: {e}")
        traceback.print_exc()
        return "error", 0, "error"


def main():
    print("  Irish S.I. PDF Processing")
    
    try:
        print(f"\nConfiguration:")
        print(f"  Input directory:  {INPUT_DIR}")
        print(f"  Output directory: {OUTPUT_DIR}")
        print(f"  Global JSONL:     {GLOBAL_JSONL or 'None'}")
        print(f"  OCR enabled:      {ENABLE_OCR}")
        print(f"  Years to process: {YEARS_TO_PROCESS}")
        
        input_dir = Path(INPUT_DIR)
        output_dir = Path(OUTPUT_DIR)
        global_jsonl = Path(GLOBAL_JSONL) if GLOBAL_JSONL else None
        enable_ocr = ENABLE_OCR
        
        # Validate input directory
        if not input_dir.exists():
            print(f"\n ERROR: Input directory does not exist: {input_dir.absolute()}")
            sys.exit(1)
        
        if not input_dir.is_dir():
            print(f"\n ERROR: Input path is not a directory: {input_dir.absolute()}")
            sys.exit(1)
        
        # Validate OCR availability
        if enable_ocr and not OCR_AVAILABLE:
            print("\n WARNING: OCR requested")
            print(" Install with: pip install pytesseract Pillow pdfplumber")
            enable_ocr = False
        
        stop_words, lemmatizer = initialize_text_tools()
        
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f" Output directory ready: {output_dir.absolute()}")
        
        if global_jsonl:
            if global_jsonl.exists():
                global_jsonl.unlink()
        
        overall_stats = {"docs": 0, "sections": 0, "ocr_used": 0, "errors": 0}
        
        for year in YEARS_TO_PROCESS:
            year_input_dir = input_dir / str(year)
            year_output_dir = output_dir / str(year)
            
            if not year_input_dir.exists():
                print(f"\nSkipping year {year}: folder not found at {year_input_dir}")
                continue
            
            pdf_paths = list(year_input_dir.glob("*.pdf"))
            
            if not pdf_paths:
                print(f"\nSkipping year {year}: no PDF files found")
                continue
            
            print(f"  PROCESSING YEAR: {year}")
            print(f"  Input:  {year_input_dir}")
            print(f"  Output: {year_output_dir}")
            print(f"  Files:  {len(pdf_paths)} PDFs")
            
            year_output_dir.mkdir(parents=True, exist_ok=True)
            
            year_stats = {"docs": 0, "sections": 0, "ocr_used": 0, "errors": 0}
            
            for pdf in sorted(pdf_paths):
                doc_id, n_sections, method = process_pdf(
                    pdf, year_output_dir, global_jsonl, enable_ocr, stop_words, lemmatizer
                )
                
                if doc_id != "error":
                    year_stats["docs"] += 1
                    year_stats["sections"] += n_sections
                    overall_stats["docs"] += 1
                    overall_stats["sections"] += n_sections
                    if method == "ocr":
                        year_stats["ocr_used"] += 1
                        overall_stats["ocr_used"] += 1
                else:
                    year_stats["errors"] += 1
                    overall_stats["errors"] += 1
            
            print(f"\n Year {year} complete:")
            print(f" Documents: {year_stats['docs']}")
            print(f" Sections:  {year_stats['sections']}")
            print(f" Errors:    {year_stats['errors']}")
        
        print(f" ALL PROCESSING COMPLETE")
        print(f" Total documents:  {overall_stats['docs']}")
        print(f" Total sections:   {overall_stats['sections']}")
        print(f" OCR used:         {overall_stats['ocr_used']}")
        print(f" Errors:           {overall_stats['errors']}")
        print(f" Output location:  {output_dir.absolute()}")
        if global_jsonl:
            print(f"  Global JSONL:     {global_jsonl.absolute()}")
        print(f"="*70 + "\n")
        
    except KeyboardInterrupt:
        print(f"\n\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n FATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()