import torch
import os
import random
import json
from pathlib import Path
from tqdm import tqdm
import sys
import math
from tabulate import tabulate


# =====================================================
# COLOR UTILITIES
# =====================================================
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def format_ppl_delta(delta_value):
    """Format PPL delta with color: negative (improvement) in green, positive in red"""
    if delta_value == 'N/A' or isinstance(delta_value, str):
        return delta_value
    # Negative delta = improvement (PPL went down) = GREEN
    # Positive delta = worsening (PPL went up) = RED
    if delta_value < 0:
        return f"{Colors.GREEN}{delta_value:+.2f}{Colors.RESET}"
    else:
        return f"{Colors.RED}{delta_value:+.2f}{Colors.RESET}"


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
from data.preProcessed.ytcom import YTcommentsDataset
from data.preProcessed.wiki import WikiDataset

from data.preProcessed.simpwiki import simpleWikiDataset
from data.preProcessed.webcrawl import WebCrawlDataset
from data.preProcessed.guten import GutenWikiDataset
from data.preProcessed.fanfic import FanFicDataset
from data.preProcessed.flexible import FlexibleDataset
from data.preProcessed.flexible_lines import FlexiblLineseDataset

from data.processed.alpacaInst import AlpacaDataset
from data.processed.casualConversation import CasualConversationDataset
from data.processed.chatAlpaca20k import ChatAlpacaDataset
from data.processed.personaChat import PersonaChatDataset


# =====================================================
# UTILITIES
# =====================================================
def print_epoch_summary(epoch, train_metrics, val_metrics, tokens_seen=0, prev_val_ppl=None, initial_val_ppl=None):
    """
    Print epoch metrics in a formatted table
    
    train_metrics/val_metrics format: 
    {'Dataset_A': {'loss': 0.5, 'ppl': 10.2}, 'Dataset_B': {...}}
    tokens_seen: total tokens processed so far
    prev_val_ppl: dict of previous val PPLs for delta calculation
    initial_val_ppl: dict of initial val PPLs for net change calculation
    """
    table_data = []
    prev_val_ppl = prev_val_ppl or {}
    initial_val_ppl = initial_val_ppl or {}
    
    # Get all unique dataset names
    datasets = sorted(list(set(train_metrics.keys()) | set(val_metrics.keys())))
    
    for ds in datasets:
        t_loss = train_metrics.get(ds, {}).get('loss', 'N/A')
        v_loss = val_metrics.get(ds, {}).get('loss', 'N/A')
        
        # Calculate Perplexity (exp of cross-entropy loss)
        t_ppl = torch.exp(torch.tensor(t_loss)).item() if isinstance(t_loss, float) else 'N/A'
        v_ppl = torch.exp(torch.tensor(v_loss)).item() if isinstance(v_loss, float) else 'N/A'
        
        # Calculate PPL delta (from previous checkpoint)
        ppl_delta = 'N/A'
        if isinstance(v_ppl, float) and ds in prev_val_ppl:
            delta = v_ppl - prev_val_ppl[ds]
            ppl_delta = format_ppl_delta(delta)
        
        # Calculate net change (from initial checkpoint)
        net_change = 'N/A'
        if isinstance(v_ppl, float) and ds in initial_val_ppl:
            change = v_ppl - initial_val_ppl[ds]
            net_change = format_ppl_delta(change)
        
        table_data.append([
            ds, 
            f"{t_loss:.4f}" if isinstance(t_loss, float) else t_loss,
            f"{v_loss:.4f}" if isinstance(v_loss, float) else v_loss,
            f"{t_ppl:.2f}" if isinstance(t_ppl, float) else t_ppl,
            f"{v_ppl:.2f}" if isinstance(v_ppl, float) else v_ppl,
            ppl_delta,
            net_change
        ])

    headers = ["Dataset", "Train Loss", "Val Loss", "Train PPL", "Val PPL", "PPL Δ", "Net Δ"]
    
    print(f"\n--- Step {epoch} | Tokens Seen: {tokens_seen:,} ---")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print("-" * 40)


class ShuffleIndexer:
    """Efficient batch indexing with shuffling"""
    def __init__(self, data_len, block_size):
        self.n = data_len - block_size - 1
        self.indices = torch.randperm(self.n).tolist()
        self.ptr = 0

    def next_indices(self, batch_size):
        # If we don't have enough indices left, reshuffle
        if self.ptr + batch_size > self.n:
            self.indices = torch.randperm(self.n).tolist()
            self.ptr = 0
        
        batch_idx = self.indices[self.ptr : self.ptr + batch_size]
        self.ptr += batch_size
        return torch.tensor(batch_idx)


class TrainingConfig:
    # -----------------------
    # Model
    # -----------------------
    n_embd = 768  # CHANGED: safe high-capacity setting for 6GB VRAM target
    n_head = 8
    n_kv_head = 2
    n_layer = 16  # CHANGED: pairs with n_embd=768
    block_size = 1024
    dropout = 0.1

    # -----------------------
    # Training
    # -----------------------
    batch_size = 4
    grad_accum_steps = 10
    
    max_iters = 6000
    total_iters = 8000 #the imaginary end point (total * grad_accum_steps)
    warmup_iters = 1200
    eval_interval = 500

    learning_rate = 3e-4
    min_lr = 5e-5
    # LR modifications
    lr_scale_on_resume = 1.4e-4 # absolute LR to set on resume when override_on_resume is True
    override_on_resume = False # True: resume from lr_scale_on_resume, False: keep checkpoint LR
    decay_stretch = 1.6
    
    weight_decay = 0.01
    grad_clip = 0.8

    eval_iters = 15
    checkpoint_dir = "./checkpoints2"

    use_gradient_checkpointing = True
    use_torch_compile = False
    use_tf32 = True
    use_8bit_adam = True
    use_fused_adamw = True

class InteractConfig(TrainingConfig):
    checkpoint_dir = "./checkpoints/Interact"
    learning_rate = 3e-5
    min_lr = 5e-6
    
    warmup_iters = 100
    max_iters = 6000
    total_iters = 6000
    
    dropout = 0.05
    weight_decay = 0.0
    grad_clip = 1.0

    batch_size = 4
    grad_accum_steps = 10

# =====================================================
# TRAINER
# =====================================================
class Trainer:
    def _build_optimizer(self, use_fused_adamw=False, use_8bit_adam=False):
        if use_8bit_adam:
            if self.device.type != "cuda":
                raise RuntimeError("8-bit Adam requires CUDA")
            try:
                import bitsandbytes as bnb
            except ImportError as e:
                raise RuntimeError("bitsandbytes is not available") from e

            return bnb.optim.AdamW8bit(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )

        optimizer_kwargs = {
            "lr": self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
            "foreach": False,
        }

        if use_fused_adamw and self.device.type == "cuda":
            optimizer_kwargs["fused"] = True

        return torch.optim.AdamW(self.model.parameters(), **optimizer_kwargs)

    def _enforce_optimizer_backend_flags(self):
        if getattr(self, "using_8bit_adam", False):
            return

        for group in self.optimizer.param_groups:
            group["foreach"] = False
            if self.fused_adamw_active and self.device.type == "cuda":
                group["fused"] = True
            elif "fused" in group:
                group["fused"] = False

    def _move_optimizer_state_to_device(self):
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(self.device)

    def _apply_lr_controls(self, resumed_from_checkpoint=False):
        """Optionally override resumed LR to a target absolute value and keep scheduler progression."""
        if not resumed_from_checkpoint:
            return

        if not bool(getattr(self.config, "override_on_resume", False)):
            return

        target_lr = float(getattr(self.config, "lr_scale_on_resume", 0.0))
        if target_lr <= 0:
            print(f"⚠ Ignoring non-positive lr_scale_on_resume={target_lr}")
            return

        current_lr = float(self.optimizer.param_groups[0]["lr"])
        if current_lr <= 0:
            print(f"⚠ Cannot override LR because current checkpoint LR is non-positive: {current_lr}")
            return

        scale = target_lr / current_lr

        if abs(scale - 1.0) < 1e-12:
            return

        for group in self.optimizer.param_groups:
            group["lr"] *= scale

        if hasattr(self.scheduler, "base_lrs"):
            self.scheduler.base_lrs = [lr * scale for lr in self.scheduler.base_lrs]
        if hasattr(self.scheduler, "_last_lr"):
            self.scheduler._last_lr = [lr * scale for lr in self.scheduler._last_lr]

        self.config.learning_rate *= scale
        self.config.min_lr *= scale
        print(
            f"✓ Applied resume LR override: target={target_lr:.2e} | "
            f"current lr={self.optimizer.param_groups[0]['lr']:.2e}"
        )

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

        # Optional TF32 acceleration toggle (faster matmuls on supported NVIDIA GPUs)
        if self.device.type == "cuda":
            use_tf32 = bool(getattr(config, "use_tf32", False))
            torch.backends.cuda.matmul.allow_tf32 = use_tf32
            torch.backends.cudnn.allow_tf32 = use_tf32
            torch.set_float32_matmul_precision("high" if use_tf32 else "highest")
            print(f"TF32 {'enabled' if use_tf32 else 'disabled'}")

        os.makedirs(config.checkpoint_dir, exist_ok=True)

        # Tokenizer & model
        self.tokenizer = BaseTokenizer()
        self.model = LuminousLM(
            vocab_size=self.tokenizer.vocab_size,
            n_embd=config.n_embd,
            n_head=config.n_head,
            n_kv_head=getattr(config, "n_kv_head", None),
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
                print("✓ Model compiled")
            except Exception as e:
                print(f"⚠ Could not compile model: {e}")
                print("  Continuing without compilation (this is fine)")

        print(f"Model parameters: {self.model.get_num_params():,}")

        # Optimizer
        use_8bit_adam = bool(getattr(config, "use_8bit_adam", False))
        use_fused_adamw = bool(getattr(config, "use_fused_adamw", False)) and self.device.type == "cuda"
        self.using_8bit_adam = False
        self.fused_adamw_active = False

        if use_8bit_adam and self.device.type != "cuda":
            print("⚠ 8-bit Adam requested but CUDA is unavailable; falling back to standard AdamW")
            use_8bit_adam = False

        try:
            self.optimizer = self._build_optimizer(
                use_fused_adamw=use_fused_adamw and not use_8bit_adam,
                use_8bit_adam=use_8bit_adam,
            )
            self.using_8bit_adam = use_8bit_adam
            self.fused_adamw_active = use_fused_adamw and not use_8bit_adam
            if self.using_8bit_adam:
                print("✓ bitsandbytes AdamW8bit enabled")
            elif self.fused_adamw_active:
                print("✓ Fused AdamW enabled")
        except (TypeError, RuntimeError) as e:
            if use_8bit_adam:
                print(f"⚠ 8-bit Adam unavailable ({e}); falling back to torch AdamW")
                self.optimizer = self._build_optimizer(use_fused_adamw=use_fused_adamw, use_8bit_adam=False)
                self.using_8bit_adam = False
                self.fused_adamw_active = use_fused_adamw
                if self.fused_adamw_active:
                    print("✓ Fused AdamW enabled")
            elif use_fused_adamw:
                print(f"⚠ Fused AdamW unavailable ({e}); falling back to standard AdamW")
                self.optimizer = self._build_optimizer(use_fused_adamw=False, use_8bit_adam=False)
                self.fused_adamw_active = False
            else:
                raise

        self._enforce_optimizer_backend_flags()

        # =====================================================
        # 🔥 FIXED LR SCHEDULER (warmup + cosine decay)
        # =====================================================
        decay_stretch = max(1e-6, float(getattr(config, "decay_stretch", 1.0)))
        decay_span = max(1, int((config.total_iters - config.warmup_iters) * decay_stretch))
        if decay_stretch != 1.0:
            print(
                f"LR decay stretch active: x{decay_stretch:.2f} "
                f"(decay span {config.total_iters - config.warmup_iters} -> {decay_span} steps)"
            )

        def lr_lambda(step):
            # step = optimizer step count
            if step < config.warmup_iters:
                return step / max(1, config.warmup_iters)

            progress = (step - config.warmup_iters) / max(
                1, decay_span
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
        self.tokens_seen = 0  # Track cumulative tokens
        self.prev_val_ppl = {}  # Track previous val PPLs for delta
        self.initial_val_ppl = {}  # Track starting PPLs for net change calculation
        
        # Initialize ShuffleIndexers for each dataset/split
        self.indexers = {}
        for dataset_name, dataset in self.datasets.items():
            for split in ["train", "val"]:
                data = dataset.train_data if split == "train" else dataset.val_data
                if len(data) > self.config.block_size:
                    self.indexers[f"{dataset_name}_{split}"] = ShuffleIndexer(
                        len(data), self.config.block_size
                    )

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
        
        # Use ShuffleIndexer for efficient batching
        indexer_key = f"{dataset_name}_{split}"
        if indexer_key in self.indexers:
            ix = self.indexers[indexer_key].next_indices(self.config.batch_size)
        else:
            # Fallback to random if indexer doesn't exist
            ix = torch.randint(
                len(data) - self.config.block_size - 1,
                (self.config.batch_size,)
            )
        
        x = torch.stack([data[i:i+self.config.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.config.block_size+1] for i in ix])

        if mask is not None:
            m = torch.stack([mask[i+1:i+self.config.block_size+1] for i in ix])
            y = y.clone()
            y[m == 0] = -100
        
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
                mask = getattr(dataset, "train_mask" if split == "train" else "val_mask", None)
                
                if len(data) <= self.config.block_size:
                    losses[f"{split}_{dataset_name}"] = float("inf")
                    continue

                dataset_losses = []
                for _ in range(self.config.eval_iters):
                    # Use ShuffleIndexer for per-dataset evaluation
                    indexer_key = f"{dataset_name}_{split}"
                    if indexer_key in self.indexers:
                        ix = self.indexers[indexer_key].next_indices(self.config.batch_size)
                    else:
                        ix = torch.randint(len(data) - self.config.block_size - 1, (self.config.batch_size,))
                    
                    x = torch.stack([data[i:i+self.config.block_size] for i in ix]).to(self.device)
                    y = torch.stack([data[i+1:i+self.config.block_size+1] for i in ix]).to(self.device)
                    
                    # ✅ FIX: Apply mask in per-dataset evaluation
                    if mask is not None:
                        m = torch.stack([mask[i+1:i+self.config.block_size+1] for i in ix]).to(self.device)
                        y = y.clone()
                        y[m == 0] = -100
                    
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
        # prompt="### System: You are Lumi. Answer my request below.\n## User: What is the name of our planet?\n#### Response:",
        prompt = "zzwho the fuck do you think you are? Do you know how",
        max_new_tokens=50,
        min_new_tokens=30,
        temperature=0.8,
        top_k=50
    ):
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
            
            # Stop if we hit EOT token
            if next_id.item() == self.tokenizer.eot_token:
                break

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
            print(f"No checkpoint found at {path}")
            return False

        ckpt = torch.load(path, map_location=self.device)
        load_result = self.model.load_state_dict(ckpt["model"], strict=False)
        if load_result.missing_keys or load_result.unexpected_keys:
            print("Checkpoint loaded with non-strict key matching due to architecture changes:")
            if load_result.missing_keys:
                print(f"  Missing keys: {len(load_result.missing_keys)}")
            if load_result.unexpected_keys:
                print(f"  Unexpected keys: {len(load_result.unexpected_keys)}")

        optimizer_loaded = False
        try:
            self.optimizer.load_state_dict(ckpt["optimizer"])
            optimizer_loaded = True
        except Exception as e:
            print(f"⚠ Optimizer state mismatch ({e}); keeping newly initialized optimizer state")

        if optimizer_loaded:
            self._enforce_optimizer_backend_flags()
            self._move_optimizer_state_to_device()
            self.scheduler.load_state_dict(ckpt["scheduler"])
            self.scaler.load_state_dict(ckpt["scaler"])
            self.start_iter = ckpt["iter"]
        else:
            self.start_iter = 0

        self.best_val_loss = ckpt["best_val_loss"]

        print(f"Resumed from iter {self.start_iter}")
        return True
    
    def load_model_weights(self, path):
        if not os.path.exists(path):
            print(f"Warning: Model weights not found at {path}")
            return False
            
        ckpt = torch.load(path, map_location=self.device)
        if "model" in ckpt:
            load_result = self.model.load_state_dict(ckpt["model"], strict=False)
        else:
            load_result = self.model.load_state_dict(ckpt, strict=False)
        if load_result.missing_keys or load_result.unexpected_keys:
            print("Model weights loaded with non-strict key matching due to architecture changes:")
            if load_result.missing_keys:
                print(f"  Missing keys: {len(load_result.missing_keys)}")
            if load_result.unexpected_keys:
                print(f"  Unexpected keys: {len(load_result.unexpected_keys)}")
        print(f"Loaded model weights from {path}")
        return True

    # =====================================================
    # TRAIN LOOP
    # =====================================================
    def train(self, resume=True, load_model_only=None):
        resumed_from_checkpoint = False
        if load_model_only is not None:
            self.load_model_weights(load_model_only)
            resume = False
        elif resume:
            resumed_from_checkpoint = self.load_checkpoint()

        self._apply_lr_controls(resumed_from_checkpoint=resumed_from_checkpoint)

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(
            range(self.start_iter, self.start_iter + self.config.max_iters),
            desc="Training"
        )
        
        # ✅ FIX: Track gradient accumulation separately
        accum_step = 0

        for iter_num in pbar:
            xb, yb = self.get_batch("train")
            if xb is None:
                continue

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                _, loss = self.model(xb, yb)
                loss = loss / self.config.grad_accum_steps

            self.scaler.scale(loss).backward()
            accum_step += 1

            # ✅ FIX: Use accum_step for proper gradient accumulation
            if accum_step % self.config.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip
                )
                try:
                    self.scaler.step(self.optimizer)
                except (AssertionError, RuntimeError) as e:
                    err_text = str(e)
                    is_known_fused_amp_issue = (
                        "grad_scale is None and found_inf is None" in err_text
                        or "Expected all tensors to be on the same device" in err_text
                    )
                    if self.fused_adamw_active and is_known_fused_amp_issue:
                        print("⚠ Fused AdamW + AMP backend mismatch detected; switching to standard AdamW for stability")
                        current_lr = self.optimizer.param_groups[0]["lr"]
                        last_epoch = self.scheduler.last_epoch
                        self.optimizer = self._build_optimizer(use_fused_adamw=False)
                        self.fused_adamw_active = False
                        for group in self.optimizer.param_groups:
                            group["lr"] = current_lr
                        self._enforce_optimizer_backend_flags()
                        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self.scheduler.lr_lambdas, last_epoch=last_epoch)
                        self.scaler.step(self.optimizer)
                    else:
                        raise
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                # Track tokens: batch_size × block_size per accumulation step
                self.tokens_seen += self.config.batch_size * self.config.block_size

            if iter_num % 10 == 0:
                pbar.set_postfix(
                    loss=f"{loss.item() * self.config.grad_accum_steps:.4f}",
                    lr=f"{self.scheduler.get_last_lr()[0]:.2e}",
                    tokens=f"{self.tokens_seen/1e6:.1f}M"
                )

            # ✅ FIX: Skip evaluation at iter 0
            if iter_num % self.config.eval_interval == 0 and iter_num > 0:
                losses = self.estimate_loss()
                
                # Format losses for print_epoch_summary
                train_metrics = {}
                val_metrics = {}
                
                for dataset_name in self.datasets.keys():
                    train_metrics[dataset_name] = {'loss': losses.get(f'train_{dataset_name}', float('inf'))}
                    val_metrics[dataset_name] = {'loss': losses.get(f'val_{dataset_name}', float('inf'))}
                
                # Calculate current val PPLs for next delta
                current_val_ppl = {}
                for dataset_name in self.datasets.keys():
                    loss_val = losses.get(f'val_{dataset_name}', float('inf'))
                    if isinstance(loss_val, float) and loss_val != float('inf'):
                        current_val_ppl[dataset_name] = torch.exp(torch.tensor(loss_val)).item()
                        # Track initial PPLs on first eval (per dataset)
                        if dataset_name not in self.initial_val_ppl:
                            self.initial_val_ppl[dataset_name] = current_val_ppl[dataset_name]
                
                print_epoch_summary(iter_num, train_metrics, val_metrics, self.tokens_seen*5, self.prev_val_ppl, self.initial_val_ppl)
                
                # Update prev_val_ppl for next eval
                self.prev_val_ppl = current_val_ppl
                
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

        # ✅ Save final checkpoint
        print("\nTraining complete!")
        self.save_checkpoint(self.start_iter + self.config.max_iters - 1)
        
        # ✅ Print final net PPL change summary
        if self.initial_val_ppl and self.prev_val_ppl:
            print(f"\n{Colors.BOLD}=== FINAL PPL IMPROVEMENT SUMMARY ==={Colors.RESET}")
            summary_data = []
            for dataset_name in sorted(self.initial_val_ppl.keys()):
                initial = self.initial_val_ppl.get(dataset_name, 'N/A')
                final = self.prev_val_ppl.get(dataset_name, 'N/A')
                if isinstance(initial, float) and isinstance(final, float):
                    net_change = final - initial
                    colored_change = format_ppl_delta(net_change)
                    pct_change = (net_change / initial) * 100 if initial != 0 else 0
                    summary_data.append([
                        dataset_name,
                        f"{initial:.2f}",
                        f"{final:.2f}",
                        colored_change,
                        f"{pct_change:+.1f}%"
                    ])
            headers = ["Dataset", "Initial PPL", "Final PPL", "Net Change", "% Change"]
            print(tabulate(summary_data, headers=headers, tablefmt="grid"))
            print(f"{Colors.BOLD}Training duration: {self.tokens_seen:,} tokens | {self.config.max_iters} steps{Colors.RESET}")


# =====================================================
# ENTRY POINT
# =====================================================
if __name__ == "__main__":
    config = TrainingConfig()

    run_meta_path = os.path.join(config.checkpoint_dir, "dataset_run_meta.json")

    def _load_and_bump_run_count(meta_path):
        run_count = 0
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                run_count = int(data.get("run_count", 0))
            except (OSError, ValueError, json.JSONDecodeError):
                run_count = 0

        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"run_count": run_count + 1}, f)

        return run_count

    run_count = _load_and_bump_run_count(run_meta_path)

    def auto_skip(take, base_skip=0):
        return base_skip + (run_count * take)

    print(f"Dataset run count: {run_count} | automatic skip = base_skip + run_count * take")

    # fic_ds  = FanFicDataset(skip=1000, take = 1000)
    dclm_ds = FlexiblLineseDataset("mlfoundations/dclm-baseline-1.0",skip=auto_skip(take=5000, base_skip=-40000), take=5000)
    falcon_ds = FlexiblLineseDataset("tiiuae/falcon-refinedweb",text_field="content", skip=auto_skip(take=10000, base_skip=-70000), take=10000)
    tnsfw_ds = FlexiblLineseDataset("Maxx0/Testing_new_nsfw",text_field="message", skip=auto_skip(take=10000, base_skip=-60000), take=10000)
    altnsfw_ds = FlexiblLineseDataset("mickume/alt_nsfw", skip=auto_skip(take=30000, base_skip=-180000), take=30000)
    edu_ds = FlexibleDataset("HuggingFaceFW/fineweb-edu", subset="default", skip=auto_skip(take=10000, base_skip=0), take=10000)
    #essay_ds = FlexibleDataset("qwedsacf/ivypanda-essays", text_field="TEXT", skip=auto_skip(take=3000, base_skip=0), take=3000)
    c4_ds = FourchanDataset(skip=auto_skip(take=10000, base_skip=-20000), take=10000)
    chan_ds = FlexiblLineseDataset("kjj0/4chanpol", skip=auto_skip(take=40000, base_skip=0), take=40000)
    tiny_ds = TinyStoriesDataset(skip=auto_skip(take=10000, base_skip=0), take=10000)
    #ore_ds = LightNovelDataset(series=[])
    #book_ds = BookDataset(skip=auto_skip(take=100000, base_skip=0), take=100000)
    #swik_ds = simpleWikiDataset(skip=auto_skip(take=50000, base_skip=0), take=50000)
    climb_ds = FlexiblLineseDataset("karpathy/climbmix-400b-shuffle", skip=auto_skip(take=10000, base_skip=-50000), take=10000)
    fine_ds = FlexiblLineseDataset("HuggingFaceFW/finephrase", text_field="text", subset="all", skip=auto_skip(take=10000, base_skip=-30000), take=10000)
    cos_ds = FlexibleDataset("kenhktsui/cosmopedia_quality_score_v2", subset="wikihow", skip=auto_skip(take=10000, base_skip=0), take=10000)
    stories_ds = FlexibleDataset("kenhktsui/cosmopedia_quality_score_v2", subset="stories", skip=auto_skip(take=10000, base_skip=0), take=10000)
    crawl_ds = WebCrawlDataset(skip=auto_skip(take=10000, base_skip=-40000), take=10000)
    # # # gut_ds = GutenWikiDataset(skip = 0, take=50000)
    wiki_ds = WikiDataset(skip=auto_skip(take=4000, base_skip=0), take=4000)
    reddit_ds = RedditDataset(skip=auto_skip(take=100000, base_skip=0), take=100000)
    # yt_ds = YTcommentsDataset()
    subhub_ds = SubtitlesHuggingDataset(skip=auto_skip(take=100000, base_skip=0), take=100000)
    #webnov_ds = FlexibleDataset("OmniAICreator/RoyalRoad-1.61M", skip=auto_skip(take=1500, base_skip=0), take=1500)

    # dd_ds = DailyDialogDataset()

    trainer = Trainer(config, 
                      {
                        "tiny": tiny_ds,
                        "fine": fine_ds,
                        "wikihow": cos_ds,
                        "stories": stories_ds,
                        "reddit": reddit_ds,
                        "edu": edu_ds,
                        "crawl": crawl_ds,
                        "climb": climb_ds,
                        "wiki": wiki_ds,
                        "dclm": dclm_ds,
                        # "yt": yt_ds,
                        "subhub": subhub_ds,
                        "chan": chan_ds,
                        "4c": c4_ds,
                        "tnsfw": tnsfw_ds,
                        "altnsfw": altnsfw_ds,
                        # "dd": dd_ds,
                        # "fanfic": fic_ds,
                        "falcon": falcon_ds,
                       },
                       mix_ratios={
                        "tiny": 0.02,
                        "fine": 0.08,
                        "wikihow": 0.05,
                        "stories": 0.03, 
                        "reddit": 0.05,
                        "edu": 0.1,
                        "crawl": 0.05,
                        "climb": 0.05,
                        "wiki": 0.1,
                        "dclm": 0.13,
                        # "yt": 0.06,
                        "subhub": 0.05,
                        "chan": 0.03,
                        "4c": 0.05,
                        "tnsfw": 0.06,
                        "altnsfw": 0.05,
                        # "dd": 0.05,
                        # "fanfic": 0.05,
                        "falcon": 0.1,
                       })

    trainer.train(resume=True)
    
    # config = InteractConfig()
    # alpac_ds = AlpacaDataset()

    # trainer = Trainer(config, {"alpaca": alpac_ds})

    # trainer.train(resume=False, load_model_only="./checkpoints/checkpoint_latest.pth")