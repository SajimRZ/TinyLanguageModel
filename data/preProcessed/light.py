
from pypdf import PdfReader
import re
from typing import List
from pathlib import Path
from data.preProcessed.base import BaseTokenizer

class LightNovelDataset:

    def __init__(self, series, val_split = 0.1):
        self.tokenizer = BaseTokenizer()
        self.arr = series
        self.val_split = val_split
        self.texts = []

        for i in self.arr:
            novel = PdfReader(Path.cwd()/"data"/"raw"/i)
            paragraphs = []
            for page in novel.pages:
                page_text = page.extract_text()
                paragraph = self.extract_paragraphs(page_text, min_length=30)
                paragraphs.extend(paragraph)
            paragraphs = paragraphs[4:]
            self.texts.extend(paragraphs)
            
        print(f"Extracted {len(self.texts)} paragraphs from {len(self.arr)} light novels.")

        split_idx = int((1 - val_split) * len(self.texts))
        self.train_data = self.tokenizer.tokenize_texts(self.texts[:split_idx])
        self.val_data = self.tokenizer.tokenize_texts(self.texts[split_idx:])

        print(f"Train tokens: {len(self.train_data):,}")
        print(f"Val tokens: {len(self.val_data):,}")




    def normalize_quotes(self,text: str) -> str:
        replacements = {
            "“": '"',
            "”": '"',
            "‘": "'",
            "’": "'",
            ",": ","
        }
        for fancy, standard in replacements.items():
            text = text.replace(fancy, standard)
        return text


    def extract_paragraphs(self,text: str, min_length: int = 30) -> List[str]:
        # Normalize all quotes first
        text = self.normalize_quotes(text)
        
        # Remove page numbers and other metadata (common in PDFs)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Split into paragraphs (separated by blank lines)
        paragraphs = re.split(r'\n\s*\n+', text)
        
        # Clean and filter paragraphs
        cleaned_paragraphs = []
        for para in paragraphs:
            # Remove extra whitespace and join lines
            cleaned = ' '.join(para.split())
            
            # Skip short paragraphs (likely headers, page numbers, etc.)
            if len(cleaned) >= min_length:
                
                cleaned_paragraphs.append(cleaned)
        
        return cleaned_paragraphs


# def normalize_quotes(text: str) -> str:
#     replacements = {
#         "“": '"',
#         "”": '"',
#         "‘": "'",
#         "’": "'",
#         ",": ","
#     }
#     for fancy, standard in replacements.items():
#         text = text.replace(fancy, standard)
#     return text


# def extract_paragraphs(text: str, min_length: int = 30) -> List[str]:
#     # Normalize all quotes first
#     text = normalize_quotes(text)
    
#     # Remove page numbers and other metadata (common in PDFs)
#     text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
    
#     # Split into paragraphs (separated by blank lines)
#     paragraphs = re.split(r'\n\s*\n+', text)
    
#     # Clean and filter paragraphs
#     cleaned_paragraphs = []
#     for para in paragraphs:
#         # Remove extra whitespace and join lines
#         cleaned = ' '.join(para.split())
#         cleaned = re.sub(r'^\d+[\.\)\:]?\s+', '', cleaned)
#         # Skip short paragraphs (likely headers, page numbers, etc.)
#         if len(cleaned) >= min_length:
#             cleaned_paragraphs.append(cleaned)
    
#     return cleaned_paragraphs

# if __name__ == "__main__":
#     cur_dir = Path.cwd()
#     pdf = cur_dir/"data"/"raw"/"rascal1.pdf"
#     df = PdfReader(cur_dir/"data"/"raw"/pdf)
#     paragraphs = []
#     for pages in df.pages:
#         page_text = pages.extract_text()
#         paragraph = extract_paragraphs(page_text, min_length=30)
        
#         paragraphs.extend(paragraph)
#     paragraphs = paragraphs[4:]

    
#     # Show the list structure
#     print("\n\nAs a Python list:")
#     print(f"Type: {type(paragraphs)}")
#     print(f"Length: {len(paragraphs)}")
#     for i in range(15):
#         print(paragraphs[i])
#         print("==========================")
#         print(len(paragraphs[i]))
#         print("==========================")
#         print("==========================")
#     # print(paragraphs[13:15])
