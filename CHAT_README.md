# Lumi Chat System

A terminal-based interactive chat interface for the Luminous Language Model.

## Quick Start

### Option 1: Simple Launcher (Recommended)

```bash
python run_chat.py
```

### Option 2: Full Control

```bash
python -m interface.chat
```

### Option 3: Single Message

```bash
python -m interface.chat --message "Hello Lumi!"
```

## Features

✨ **Interactive Chat** - Have natural conversations with Lumi

- Maintains conversation history for better context
- Supports multi-turn dialogues
- Intelligent sampling (temperature, top-k, top-p)

🎯 **Smart Generation**

- Uses RoPE (Rotary Position Embeddings)
- Flash Attention for efficiency
- Gradient checkpointing support
- Temperature and nucleus sampling

💾 **Checkpoint Loading**

- Auto-detects best model checkpoint
- Supports custom checkpoint paths
- Multiple fallback options

🔧 **Customizable Parameters**

- Adjustable max generation length
- Temperature control for creativity
- Top-k and top-p sampling parameters
- Device selection (CPU/GPU)

## Commands During Chat

While in the interactive chat session, use these commands:

| Command          | Effect                             |
| ---------------- | ---------------------------------- |
| `exit` or `quit` | End the chat session               |
| `clear`          | Clear conversation history         |
| `history`        | View conversation history          |
| `temp <value>`   | Set temperature (e.g., `temp 0.5`) |
| `topk <value>`   | Set top-k (e.g., `topk 40`)        |
| (any text)       | Send message to Lumi               |

## Usage Examples

### Basic Chat (Interactive Checkpoint Selection)

```bash
python run_chat.py
```

The system will show available checkpoints and let you choose:

```
📂 Available checkpoints:
  1. ./checkpoints/Interact/model_best.pth
  2. ./checkpoints/checkpoint_latest.pth
  3. ./BackupCheckpoints/attempt8.pth

Select checkpoint number (or 0 to enter custom path): 1
```

Then type your messages naturally!

### Using Specific Checkpoint

```bash
python -m interface.chat --checkpoint checkpoints/model_best.pth
```

### Single Response (No Interactive Mode)

```bash
python -m interface.chat --message "What's your favorite color?"
```

### Control Generation Parameters

```bash
python -m interface.chat --checkpoint ./checkpoints/Interact/model_best.pth --temperature 0.6 --top-k 40
```

Or during interactive chat:

```
You: temp 0.5
🌡️  Temperature set to 0.5

You: topk 40
🎯 Top-K set to 40
```

### Force CPU (if GPU issues)

```bash
python -m interface.chat --device cpu
```

## Generation Parameters Explained

- **temperature** (0.0 - 2.0):
  - Lower = More focused/deterministic
  - Higher = More creative/random
  - Default: 0.8 (matches train3.py)

- **top-k** (positive int):
  - Only sample from top-k most likely tokens
  - Default: 50 (matches train3.py)

- **max-length** (tokens):
  - Maximum number of tokens to generate
  - Default: 100

## Message Format

The chat system uses a structured message format matching train3.py:

```
### System: You are Lumi. [system prompt]
## User: What is 2+2?
#### Response: [model generates response here]
```

This format enables:

- Clear separation of roles (System/User/Response)
- Proper instruction tuning
- Better context understanding
- Consistency with training format

## Lumi's Personality

Loaded from `personality/system_prompt.txt`:

- Easygoing, whimsical, and humorous
- Energetic, spontaneous, and unpredictable
- Likes to joke and tease
- Speaks like a natural human, not an assistant
- Avoids long technical explanations

## Model Details

- **Architecture**: Luminous - Modern transformer with:
  - RoPE embeddings
  - Flash Attention
  - SwiGLU feed-forward layers
  - RMS Normalization
- **Size**: Approximately 30-40M parameters
  - Embedding dim: 512
  - Heads: 8
  - Layers: 10
  - Block size: 512

- **Training**: Instruction-tuned on diverse dialogue data
  - Reddit conversations
  - Daily dialogues
  - Light novels
  - Wikipedia
  - YouTube comments
  - And more!

## Checkpoints

When you run the chat system, you'll be prompted to select from available checkpoints:

- `checkpoints/Interact/model_best.pth` - Best interactive/instruction-tuned model
- `checkpoints/Interact/checkpoint_latest.pth` - Latest interactive checkpoint
- `checkpoints/checkpoint_latest.pth` - Latest base checkpoint
- `BackupCheckpoints/` - Various backup checkpoints

You can also specify a checkpoint directly:

```bash
python -m interface.chat --checkpoint ./checkpoints/Interact/model_best.pth
```

## Troubleshooting

### "No checkpoint found"

Place a trained model checkpoint in the `checkpoints/` directory.

### Model very slow / Memory issues

- Use `--device cpu` if GPU memory is limited
- Reduce `--max-length` parameter
- Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`

### Tokenizer errors

Ensure `data/preProcessed/base.py` is properly configured with correct vocab.

### Responses are nonsensical

- The model needs a trained checkpoint. Untrained models produce random text.
- Try adjusting `--temperature` (use 0.5 for more coherent output)

## Advanced: Custom Implementation

```python
from interface.chat import LumiChat

# Create chat instance (will prompt for checkpoint)
lumi = LumiChat()

# Or specify checkpoint directly
lumi = LumiChat(checkpoint_path='./checkpoints/Interact/model_best.pth', device='cuda')

# Single response
response = lumi.chat("Hello!", temperature=0.8, top_k=50)
print(response)

# Or full interaction
lumi.interactive_chat(temperature=0.8, top_k=50)
```

## Architecture Overview

```
User Input
    ↓
Tokenization (BPE/SentencePiece)
    ↓
Token Embedding
    ↓
[Transformer Block × 10]
  - RoPE Position Embedding
  - Multi-Head Attention (Flash Attention)
  - SwiGLU Feed-Forward
  - RMS Normalization
    ↓
Output Projection
    ↓
Sampling (Top-k, Top-p, Temperature)
    ↓
Decoding
    ↓
User Output
```

Enjoy chatting with Lumi! 🌟
