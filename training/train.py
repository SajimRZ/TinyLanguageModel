import torch
import os
import random
from pathlib import Path
from tqdm import tqdm
import sys
import math

# =====================================================
# PATH SETUP
# =====================================================
sys.path.append(str(Path(__file__).parent.parent))

from models.model import LuminousLM
from data.preProcessed.base import BaseTokenizer
from data.preProcessed.tiny import TinyStoriesDataset
from data.preProcessed.dailyDialouges import DailyDialogDataset
from data.preProcessed.sub import SubtitleDataset
from data.preProcessed.light import LightNovelDataset
from data.preProcessed.reddit import RedditDataset
from data.preProcessed.bookcore import BookDataset
from data.preProcessed.subhub import SubtitlesHuggingDataset
from data.preProcessed.chan import FourchanDataset


# =====================================================
# TRAINING CONFIG
# =====================================================
class TrainingConfig:
    # -----------------------
    # Model
    # -----------------------
    n_embd = 448
    n_head = 7
    n_layer = 7
    block_size = 512
    dropout = 0.1

    # -----------------------
    # Training
    # -----------------------
    batch_size = 12
    grad_accum_steps = 4
    max_iters = 6000
    total_iters = 60000
    warmup_iters = 2000
    eval_interval = 300

    learning_rate = 4.5e-4
    min_lr = 4.5e-5
    weight_decay = 0.01
    grad_clip = 1.0

    eval_iters = 20
    checkpoint_dir = "./checkpoints2"

    use_gradient_checkpointing = True
    use_torch_compile = False

class InteractConfig(TrainingConfig):
    checkpoint_dir = "./checkpoints/Interact"
    learning_rate = 3e-5
    min_lr = 5e-6
    
    warmup_iters = 100
    max_iters = 8000
    total_iters = 8000
    
    dropout = 0.05
    weight_decay = 0.0
    grad_clip = 1.0

    batch_size = 8
    grad_accum_steps = 3

# =====================================================
# TRAINER
# =====================================================
class Trainer:
    def __init__(self, config, datasets, mix_ratios=None):
        self.config = config
        self.datasets = datasets
        self.mix_ratios = mix_ratios or {k: 1.0 for k in datasets}

        total = sum(self.mix_ratios.values())
        self.mix_ratios = {k: v / total for k, v in self.mix_ratios.items()}

        # Device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Using device: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print("Using device: CPU")

        os.makedirs(config.checkpoint_dir, exist_ok=True)

        # Tokenizer & model
        self.tokenizer = BaseTokenizer()
        self.model = LuminousLM(
            vocab_size=self.tokenizer.vocab_size,
            n_embd=config.n_embd,
            n_head=config.n_head,
            n_layer=config.n_layer,
            block_size=config.block_size,
            dropout=config.dropout
        ).to(self.device)

        # Enable gradient checkpointing
        if hasattr(config, 'use_gradient_checkpointing') and config.use_gradient_checkpointing:
            self.model.gradient_checkpointing = True
            print("✓ Gradient checkpointing enabled (saves ~1.5GB VRAM)")
        
        # Compile model with torch.compile (PyTorch 2.0+)
        if hasattr(config, 'use_torch_compile') and config.use_torch_compile:
            try:
                print("Compiling model with torch.compile()...")
                self.model = torch.compile(
                    self.model,
                    mode='reduce-overhead'  # Options: 'default', 'reduce-overhead', 'max-autotune'
                )
                print("✓ Model compiled (expect ~20-30% speedup)")
            except Exception as e:
                print(f"⚠ Could not compile model: {e}")
                print("  Continuing without compilation (this is fine)")

        print(f"Model parameters: {self.model.get_num_params():,}")

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # =====================================================
        # 🔥 FIXED LR SCHEDULER (warmup + cosine decay)
        # =====================================================
        def lr_lambda(step):
            # step = optimizer step count
            if step < config.warmup_iters:
                return step / max(1, config.warmup_iters)

            progress = (step - config.warmup_iters) / max(
                1, config.total_iters - config.warmup_iters
            )
            progress = min(progress, 1.0)

            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return cosine * (1 - config.min_lr / config.learning_rate) + (
                config.min_lr / config.learning_rate
            )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda
        )

        # AMP
        self.use_amp = torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.start_iter = 0
        self.best_val_loss = float("inf")

    # =====================================================
    # DATA
    # =====================================================
    def get_batch(self, split="train"):
        dataset_name = random.choices(
            list(self.datasets.keys()),
            weights=list(self.mix_ratios.values())
        )[0]

        dataset = self.datasets[dataset_name]
        if split == "train":
            data = dataset.train_data
            mask = getattr(dataset, "train_mask", None)
        else:
            data = dataset.val_data
            mask = getattr(dataset, "val_mask", None)

        # if len(data) <= self.config.block_size:
        #     return None, None

        ix = torch.randint(
            len(data) - self.config.block_size-1,
            (self.config.batch_size,)
        )

        x = torch.stack([data[i:i+self.config.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.config.block_size+1] for i in ix])

        if mask is not None:
            m = torch.stack([mask[i+1:i+self.config.block_size+1] for i in ix])

            y = y.clone()
            y[m==0] = -100

        
        return x.to(self.device), y.to(self.device)

    # =====================================================
    # EVALUATION
    # =====================================================
    @torch.no_grad()
    def estimate_loss(self):
        self.model.eval()
        losses = {}

        for split in ["train", "val"]:
            # -------------------
            # Overall split loss
            # -------------------
            split_losses = []
            for _ in range(self.config.eval_iters):
                xb, yb = self.get_batch(split)
                if xb is None:
                    continue
                _, loss = self.model(xb, yb)
                split_losses.append(loss.item())

            losses[split] = (
                sum(split_losses) / len(split_losses)
                if split_losses else float("inf")
            )

            # -------------------
            # Per-dataset losses
            # -------------------
            for dataset_name, dataset in self.datasets.items():
                data = dataset.train_data if split == "train" else dataset.val_data
                if len(data) <= self.config.block_size:
                    losses[f"{split}_{dataset_name}"] = float("inf")
                    continue

                dataset_losses = []
                for _ in range(self.config.eval_iters):
                    ix = torch.randint(len(data) - self.config.block_size, (self.config.batch_size,))
                    x = torch.stack([data[i:i+self.config.block_size] for i in ix]).to(self.device)
                    y = torch.stack([data[i+1:i+self.config.block_size+1] for i in ix]).to(self.device)
                    _, loss = self.model(x, y)
                    dataset_losses.append(loss.item())

                losses[f"{split}_{dataset_name}"] = sum(dataset_losses) / len(dataset_losses)

        self.model.train()
        return losses

    # =====================================================
    # GENERATION
    # =====================================================
    @torch.no_grad()
    def generate_sample(
        self,
        prompt= 'Hey, Long time ',
        max_new_tokens=100,
        min_new_tokens=20,
        temperature=0.8,
        top_k=50
    ):
        self.model.eval()

        ids = torch.tensor(
            self.tokenizer.tokenize_texts([prompt]),
            device=self.device
        )
        if ids[-1] == self.tokenizer.eot_token:
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

        text = self.tokenizer.decode(ids[0].tolist())
        self.model.train()
        return text

    # =====================================================
    # CHECKPOINTS
    # =====================================================
    def save_checkpoint(self, iter_num):
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "iter": iter_num,
            "best_val_loss": self.best_val_loss
        }, os.path.join(self.config.checkpoint_dir, "checkpoint_latest.pth"))
        print(f"Saved checkpoint @ iter {iter_num}")

    def load_checkpoint(self):
        path = os.path.join(self.config.checkpoint_dir, "checkpoint_latest.pth")
        if not os.path.exists(path):
            return False

        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.start_iter = ckpt["iter"]
        self.best_val_loss = ckpt["best_val_loss"]

        print(f"Resumed from iter {self.start_iter}")
        return True
    
    def load_model_weights(self, path):
        ckpt = torch.load(path, map_location=self.device)
        if "model" in ckpt:
            self.model.load_state_dict(ckpt["model"])
        else:
            self.model.load_state_dict(ckpt)
        print(f"Loaded model weights from {path}")

    # =====================================================
    # TRAIN LOOP
    # =====================================================
    def train(self, resume=True, load_model_only = None):
        if load_model_only is not None:
            self.load_model_weights(load_model_only)
            resume = False
        elif resume:
            self.load_checkpoint()

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(
            range(self.start_iter, self.start_iter + self.config.max_iters),
            desc="Training"
        )

        for iter_num in pbar:
            xb, yb = self.get_batch("train")
            if xb is None:
                continue

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                _, loss = self.model(xb, yb)
                loss = loss / self.config.grad_accum_steps

            self.scaler.scale(loss).backward()

            if (iter_num + 1) % self.config.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

            if iter_num % 10 == 0:
                pbar.set_postfix(
                    loss=f"{loss.item() * self.config.grad_accum_steps:.4f}",
                    lr=f"{self.scheduler.get_last_lr()[0]:.2e}"
                )

            if iter_num % self.config.eval_interval == 0:
                losses = self.estimate_loss()
                print(f"\nEval @ {iter_num}: Train: {losses['train']} --- Val:{losses['val']}")
                print(f"Datasets wise:")
                for k in losses.keys():
                    if k.startswith("train_"):
                        print(f"  {k}: {losses[k]:.4f} || ", end="")
                print()
                for k in losses.keys():
                    if k.startswith("val_"):
                        print(f"  {k}: {losses[k]:.4f} || ", end="")
                print()


                sample = self.generate_sample()
                print("\n--- SAMPLE ---")
                print(sample)
                print("--------------\n")

                if losses["val"] < self.best_val_loss:
                    self.best_val_loss = losses["val"]
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(self.config.checkpoint_dir, "model_best.pth")
                    )
                    print(f"New best model saved with val loss {self.best_val_loss:.4f}")

                self.save_checkpoint(iter_num)

        print("Training complete")


# =====================================================
# ENTRY POINT
# =====================================================
if __name__ == "__main__":
    config = TrainingConfig()

    # tiny_ds = TinyStoriesDataset(skip=60000, take=10000)
    # ore_ds = LightNovelDataset([
    #                             #"oregairu1.pdf","oregairu2.pdf","oregairu3.pdf","oregairu4.pdf","oregairu5.pdf","oregairu6.pdf","oregairu7.pdf",
    #                             "ngnl1.pdf","ngnl2.pdf","ngnl3.pdf","ngnl4.pdf","ngnl5.pdf","ngnl6.pdf","ngnl7.pdf",
    #                             "oregairu9.pdf","oregairu10.pdf","oregairu11.pdf","oregairu12.pdf","oregairu13.pdf","oregairu14.pdf",

    #                             #"alya1.pdf","alya2.pdf","alya3.pdf","alya4.pdf","alya5.pdf","alya6.pdf","alya7.pdf",

    #                             "exstep1.pdf","exstep2.pdf","exstep3.pdf","exstep4.pdf","exstep5.pdf","exstep6.pdf","exstep7.pdf",
    #                             "exstep8.pdf","exstep9.pdf",
    #                             "kono1.pdf","kono2.pdf","kono3.pdf","kono4.pdf","kono5.pdf","kono6.pdf","kono7.pdf","kono8.pdf","kono9.pdf",

    #                             #"sis1.pdf","sis2.pdf","sis3.pdf","sis4.pdf","sis5.pdf","sis6.pdf","sis7.pdf",
    #                             "bungo1.pdf","bungo2.pdf","bungo3.pdf","bungo4.pdf","bungo8.pdf",
    #                             "monoln1.pdf","monoln2.pdf","monoln3.pdf","monoln4.pdf","monoln5.pdf","monoln6.pdf","monoln7.pdf",
    #                             "monoln8.pdf","monoln9.pdf","monoln10.pdf","monoln11.pdf","monoln12.pdf","monoln13.pdf","monoln14.pdf",

    #                             "rascal1.pdf","rascal2.pdf","rascal3.pdf","rascal4.pdf","rascal5.pdf","rascal6.pdf","rascal7.pdf",
    #                             "rascal8.pdf","rascal9.pdf","rascal10.pdf","rascal11.pdf","rascal12.pdf","rascal13.pdf","rascal14.pdf",

    #                             ])
    
    # book_ds = BookDataset(skip = 120000, take = 100000)
    # reddit_ds = RedditDataset(skip=100000, take=70000) 
    # subhub_ds =  SubtitlesHuggingDataset(skip=300000, take=200000)

    # dd_ds = DailyDialogDataset()

    # trainer = Trainer(config, 
    #                   {"tiny": tiny_ds,
    #                    "books": book_ds,
    #                    "oregairu": ore_ds,
    #                    "reddit": reddit_ds,
    #                    "subhub": subhub_ds,
    #                    "daily": dd_ds,
    #                    },
    #                    mix_ratios={
    #                        "tiny":0.05,
    #                        "books":0.17,
    #                        "oregairu":0.18,
    #                        "reddit": 0.25,
    #                        "sub_hub":0.18,
    #                        "daily": 0.17,
    #                    })

    # trainer.train(resume=True)
#21400