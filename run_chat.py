#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from interface.chat import Chat, CHECKPOINT_PATH, DEVICE

if __name__ == "__main__":
    chat = Chat(checkpoint_path=CHECKPOINT_PATH, device=DEVICE)
    chat.interactive()

