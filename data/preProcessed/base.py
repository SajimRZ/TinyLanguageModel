# Data/base.py
import tiktoken
import torch


class BaseTokenizer:
    """
    Thin wrapper around tiktoken for LM training
    """
    def __init__(self, encoding_name="gpt2"):
        self.enc = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.enc.n_vocab
        self.eot_token = self.enc.eot_token

    def tokenize_texts(self, texts):
        """
        texts: list[str]
        returns: torch.LongTensor (1D stream of tokens)
        """
        tokens = []
        for text in texts:
            if not isinstance(text, str):
                continue
            tokens.extend(self.enc.encode(text))
            tokens.append(self.eot_token)

        return torch.tensor(tokens, dtype=torch.long)
    def encode(self, text):
        return torch.tensor(self.enc.encode(text), dtype=torch.long)


    def decode(self, token_ids):
        """
        token_ids: list[int] | torch.Tensor
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.enc.decode(token_ids)
