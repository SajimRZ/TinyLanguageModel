import torch
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from models.model import LuminousLM
from data.preProcessed.base import BaseTokenizer


# =====================================================
# SETTINGS - Change these
# =====================================================
CHECKPOINT_PATH = "./checkpoints/Interact/Checkpoint_latest.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Chat:
    def __init__(self, checkpoint_path: str, device: str):
        self.device = device
        self.tokenizer = BaseTokenizer()
        
        # Model
        # Load checkpoint to get model config
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
            return
        
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        config = ckpt.get("config", {})
        
        # Create model with config from checkpoint
        self.model = LuminousLM(
            vocab_size=config.get("vocab_size", self.tokenizer.vocab_size),
            n_embd=config.get("n_embd", 768),
            n_head=config.get("n_head", 8),
            n_layer=config.get("n_layer", 16),
            block_size=config.get("block_size", 512),
            dropout=config.get("dropout", 0.0),
            n_kv_head=config.get("n_kv_head", 2),
        ).to(self.device)
        self.model.eval()
        
        # Load weights
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
            return
        
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        if "model" in ckpt:
            self.model.load_state_dict(ckpt["model"], strict=False)
        else:
            self.model.load_state_dict(ckpt, strict=False)
        
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        min_new_tokens: int = 20,
        temperature: float = 0.8,
        top_k: int = 50
    ):
        """Generate text streaming tokens one at a time"""
        self.model.eval()
        
        ids = torch.tensor(
            self.tokenizer.tokenize_texts([prompt]),
            device=self.device
        )
        if len(ids) > 0 and ids[-1] == self.tokenizer.eot_token:
            ids = ids[:-1]
        ids = ids.unsqueeze(0)
        
        generated_tokens = []
        
        for step in range(max_new_tokens):
            idx = ids[:, -self.model.block_size:]
            logits, _ = self.model(idx)
            logits = logits[:, -1, :] / temperature
            
            if step < min_new_tokens:
                logits[:, self.tokenizer.eot_token] = -1e10
            
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -1e10
            
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            ids = torch.cat([ids, next_id], dim=1)
            generated_tokens.append(next_id.item())
            
            # Decode and yield the new token
            text_chunk = self.tokenizer.decode(generated_tokens)
            yield text_chunk
            
            if next_id.item() == self.tokenizer.eot_token:
                break
        
        self.model.train()
    
    def chat(self, user_message: str) -> str:
        """Talk to Lumi"""
        prompt = f"### System: You are Lumi. Answer my request below.\n## User: {user_message}\n#### Response:"
        return self.generate(prompt, max_new_tokens=100)
    
    def interactive(self):
        """Chat loop with streaming output"""
        print("\n✨ Welcome to Lumi! ✨")
        print("Type 'exit' to quit\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ['exit', 'quit']:
                    print("👋 Goodbye!")
                    break
                
                prompt = f"### System: You are Lumi. Answer my request below.\n## User: {user_input}\n#### Response:"
                
                print("Lumi: ", end="", flush=True)
                prev_text = ""
                
                # Stream tokens and display them
                for text_chunk in self.generate(prompt, max_new_tokens=100):
                    # Only print new characters that were generated
                    new_chars = text_chunk[len(prev_text):]
                    print(new_chars, end="", flush=True)
                    prev_text = text_chunk
                
                print()  # Newline after response
                print()
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    chat = Chat(checkpoint_path=CHECKPOINT_PATH, device=DEVICE)
    chat.interactive()
