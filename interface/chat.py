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
CHECKPOINT_PATH = "./checkpoints/Interact/model_best.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Chat:
    def __init__(self, checkpoint_path: str, device: str):
        self.device = device
        self.tokenizer = BaseTokenizer()
        
        # Model
        self.model = LuminousLM(
            vocab_size=self.tokenizer.vocab_size,
            n_embd=512,
            n_head=8,
            n_layer=10,
            block_size=512,
            dropout=0.0,
        ).to(self.device)
        self.model.eval()
        
        # Load weights
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
            return
        
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        if "model" in ckpt:
            self.model.load_state_dict(ckpt["model"])
        else:
            self.model.load_state_dict(ckpt)
        
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        min_new_tokens: int = 20,
        temperature: float = 0.8,
        top_k: int = 50
    ) -> str:
        """Generate text following train3.py style"""
        self.model.eval()
        
        ids = torch.tensor(
            self.tokenizer.tokenize_texts([prompt]),
            device=self.device
        )
        if len(ids) > 0 and ids[-1] == self.tokenizer.eot_token:
            ids = ids[:-1]
        ids = ids.unsqueeze(0)
        
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
            
            if next_id.item() == self.tokenizer.eot_token:
                break
        
        text = self.tokenizer.decode(ids[0].tolist())
        self.model.train()
        return text
    
    def chat(self, user_message: str) -> str:
        """Talk to Lumi"""
        prompt = f"### System: You are Lumi. Answer my request below.\n## User: {user_message}\n#### Response:"
        return self.generate(prompt, max_new_tokens=100)
    
    def interactive(self):
        """Chat loop"""
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
                
                print("Lumi: ", end="", flush=True)
                response = self.chat(user_input)
                # Extract just the response part
                if "#### Response:" in response:
                    response = response.split("#### Response:")[-1].strip()
                # Stop at user/system markers
                for marker in ["## User:", "### System:"]:
                    if marker in response:
                        response = response.split(marker)[0].strip()
                        break
                print(response)
                print()
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    chat = Chat(checkpoint_path=CHECKPOINT_PATH, device=DEVICE)
    chat.interactive()
