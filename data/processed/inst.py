import torch
import pandas as pd
from datasets import load_dataset
import json

from data.preProcessed.base import BaseTokenizer

class InstructDataset:
    def __init__(self,skip=0, take=10000, val_split=0.1,train_mask = True):
        
        self.tokenizer = BaseTokenizer()

        tokens = []
        masks = []

        text = self._load_text()
    
    def _load_text(self):
        # try:
        #     with open()
        
        
        pass
