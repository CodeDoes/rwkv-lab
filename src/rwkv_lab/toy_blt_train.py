"""Self-contained toy training and evaluation demonstration for RWKV-7 with BLT ChannelMix.

This script trains a small, character/byte-level model using standard RWKV-7 TimeMix
and our new RWKV7BLTChannelMix block. It automatically generates a small multilingual
Tiny Stories dataset (English, French, Spanish) represented as raw UTF-8 bytes, and trains
the model, showcasing training convergence, entropy-head learning, and patch-length adaptation.
"""

from __future__ import annotations

import random
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rwkv_lab.blt_channel_mix import RWKV7BLTChannelMix
from rwkv_lab.rwkv8_deltanet import RWKV8TimeMixDeltaNet


# 1. Custom BLT-RWKV-7 Block and Model
class BLTRWKV7Block(nn.Module):
    def __init__(self, d_model: int, i: int, num_layers: int, threshold: float = 3.0, max_patch: int = 16):
        super().__init__()
        self.i = i
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        # Standard TimeMix at the byte/character level
        self.att = RWKV8TimeMixDeltaNet(
            hidden_size=d_model,
            num_heads=d_model // 16,  # head_size of 16
            head_size=16,
            layer_idx=i,
            depth_layer_id=i,
            depth_n_layer=num_layers,
            is_first_rwkv_layer=(i == 0),
            out_correct=False,
        )

        # BLT-style ChannelMix at the dynamic patch level
        self.ffn = RWKV7BLTChannelMix(
            hidden_size=d_model,
            ffn_hidden_size=d_model * 2,
            layer_idx=i,
            threshold=threshold,
            min_patch=1,
            max_patch=max_patch,
            streaming_mode="causal",
        )

    def forward(
        self,
        x: torch.Tensor,
        v_first: Optional[torch.Tensor] = None,
        initial_state_att: Optional[torch.Tensor] = None,
        shift_state_att: Optional[torch.Tensor] = None,
        shift_state_ffn: Optional[tuple] = None,
        return_state: bool = False,
        target_bytes: Optional[torch.Tensor] = None,
    ):
        # 1. TimeMix
        ln_x = self.ln1(x)
        if return_state:
            att_out, next_state_att, next_shift_att, v_first_out = self.att(
                ln_x, v_first=v_first, return_v_first=True,
                initial_state=initial_state_att, shift_state=shift_state_att, return_state=True
            )
        else:
            att_out, v_first_out = self.att(
                ln_x, v_first=v_first, return_v_first=True,
                initial_state=initial_state_att, shift_state=shift_state_att
            )
        x = x + att_out

        # 2. BLT ChannelMix
        ln_x2 = self.ln2(x)
        if return_state:
            ffn_out, next_state_ffn = self.ffn(
                ln_x2, shift_state=shift_state_ffn, return_state=True
            )
        else:
            ffn_out = self.ffn(ln_x2, shift_state=shift_state_ffn)
        x = x + ffn_out

        if return_state:
            return x, v_first_out, (next_state_att, next_shift_att), next_state_ffn

        if target_bytes is not None:
            entropy_loss = self.ffn.compute_entropy_loss(ln_x2, target_bytes)
            return x, v_first_out, entropy_loss

        return x, v_first_out


class BLTRWKV7LanguageModel(nn.Module):
    def __init__(self, vocab_size: int = 256, d_model: int = 64, n_layers: int = 2, threshold: float = 2.8, max_patch: int = 16):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            BLTRWKV7Block(d_model, i, n_layers, threshold, max_patch) for i in range(n_layers)
        ])
        self.ln_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor, states: Optional[list] = None, return_state: bool = False, target_bytes: Optional[torch.Tensor] = None):
        x = self.emb(ids)

        v_first = None
        next_states = []
        entropy_losses = []
        for i, block in enumerate(self.blocks):
            if return_state:
                init_state_att = states[i][0][0] if states else None
                shift_state_att = states[i][0][1] if states else None
                state_ffn = states[i][1] if states else None
                x, v_first, next_state_att, next_state_ffn = block(
                    x, v_first=v_first,
                    initial_state_att=init_state_att, shift_state_att=shift_state_att,
                    shift_state_ffn=state_ffn, return_state=True
                )
                next_states.append((next_state_att, next_state_ffn))
            else:
                if target_bytes is not None:
                    x, v_first, e_loss = block(x, v_first=v_first, target_bytes=target_bytes)
                    entropy_losses.append(e_loss)
                else:
                    x, v_first = block(x, v_first=v_first)

        x = self.ln_out(x)
        logits = self.head(x)

        if return_state:
            return logits, next_states
        if target_bytes is not None:
            return logits, entropy_losses
        return logits


# 2. Multilingual Tiny Stories Generator
def generate_multilingual_dataset() -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a toy multilingual dataset of simple stories in English, French, and Spanish."""
    english_stories = [
        "The cat sat on the warm mat.",
        "A small girl found a beautiful shiny key.",
        "The big dog barked happily at the bird.",
        "She drank sweet juice from a red cup.",
    ]
    french_stories = [
        "Le chat etait assis sur le tapis chaud.",
        "Une petite fille a trouve une jolie cle brillante.",
        "Le grand chien aboyait joyeusement apres l'oiseau.",
        "Elle a bu du jus sucre dans une tasse rouge.",
    ]
    spanish_stories = [
        "El gato se sento en la alfombra calida.",
        "Una niña pequeña encontro una hermosa llave brillante.",
        "El perro grande ladro alegremente al pajaro.",
        "Ella bebio un jugo dulce de una taza roja.",
    ]

    all_texts = []
    # Mix them to create a multilingual stream
    for en, fr, es in zip(english_stories, french_stories, spanish_stories):
        all_texts.extend([en, fr, es])

    # Convert all texts to UTF-8 bytes
    train_bytes = []
    for text in all_texts:
        # Wrap each story with special start and end markers
        story_bytes = [0] + list(text.encode("utf-8")) + [1]
        train_bytes.extend(story_bytes)

    # Pad/slice into sequences of length 64
    seq_len = 64
    chunks = [train_bytes[i:i + seq_len] for i in range(0, len(train_bytes) - seq_len, seq_len)]

    # Validation data (unseen stories)
    val_stories = [
        "The small bird sang a beautiful song.",
        "L'oiseau a chante une belle chanson.",
        "El pajaro canto una hermosa cancion.",
    ]
    val_bytes = []
    for text in val_stories:
        val_bytes.extend([0] + list(text.encode("utf-8")) + [1])
    while len(val_bytes) < seq_len:
        val_bytes.append(0)
    val_bytes = val_bytes[:seq_len]

    return torch.tensor(chunks, dtype=torch.long), torch.tensor([val_bytes], dtype=torch.long)


import argparse


# 3. Training Loop
def main():
    parser = argparse.ArgumentParser(description="Train a small multilingual BLT-RWKV-7 language model.")
    parser.add_argument("--d_model", type=int, default=32, help="Hidden dimension of the model")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of layers in the model")
    parser.add_argument("--threshold", type=float, default=3.5, help="Entropy threshold for patch boundary splits")
    parser.add_argument("--max_patch", type=int, default=16, help="Maximum allowed patch size in bytes")
    parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
    args = parser.parse_args()

    print("=====================================================================")
    print("        RWKV-7 + BLT ChannelMix Multilingual Toy Trainer            ")
    print("=====================================================================")

    # Hyperparameters
    d_model = args.d_model
    n_layers = args.n_layers
    threshold = args.threshold
    max_patch = args.max_patch
    lr = args.lr
    epochs = args.epochs

    # 1. Create dataset
    train_data, val_data = generate_multilingual_dataset()
    print(f"Dataset generated:")
    print(f"  Training batch shape:   {train_data.shape}")
    print(f"  Validation batch shape: {val_data.shape}")

    # 2. Build model
    model = BLTRWKV7LanguageModel(
        vocab_size=256,
        d_model=d_model,
        n_layers=n_layers,
        threshold=threshold,
        max_patch=max_patch
    )
    print(f"Model constructed with {n_layers} layers, threshold={threshold}, max_patch={max_patch}.")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Forward pass (context-aware next-byte prediction and entropy heads training)
        logits, entropy_losses = model(train_data, target_bytes=train_data) # [B, T, 256]

        # 1. Standard next-byte prediction loss
        ce_loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, 256),
            train_data[:, 1:].reshape(-1)
        )

        # 2. Entropy prediction head losses (to train the entropy predictors)
        entropy_loss = sum(entropy_losses)

        total_loss = ce_loss + 0.1 * entropy_loss
        total_loss.backward()
        optimizer.step()

        # 3. Log epoch stats and monitor average patch lengths
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_logits = model(val_data)
                val_ce = F.cross_entropy(
                    val_logits[:, :-1].reshape(-1, 256),
                    val_data[:, 1:].reshape(-1)
                ).item()

                # Calculate average patch length across layers using stored context-aware last_patch_ids
                avg_lens = []
                for b_idx, block in enumerate(model.blocks):
                    patch_ids = block.ffn.last_patch_ids
                    num_patches = int(patch_ids[0].max()) + 1
                    avg_len = val_data.shape[1] / num_patches
                    avg_lens.append(avg_len)

                lens_str = ", ".join([f"L{i}: {l:.2f}" for i, l in enumerate(avg_lens)])
                print(f"Epoch {epoch:02d} | Train CE: {ce_loss.item():.4f} | Val CE: {val_ce:.4f} | Avg Patch Lens: [{lens_str}]")
                for b_idx, block in enumerate(model.blocks):
                    ent = block.ffn.last_entropy
                    print(f"  L{b_idx} entropy: min={ent.min().item():.2f}, mean={ent.mean().item():.2f}, max={ent.max().item():.2f}")

    print("\nTraining completed successfully!")

    # 4. Multilingual Autoregressive Generation Demonstration
    print("\n--- Autoregressive Generation Demo ---")
    model.eval()
    prompt = "El gato "
    generated = list(prompt.encode("utf-8"))

    # Initialize recurrent states
    states = None

    print(f"Prompt: '{prompt}'")

    # Feed prompt to build states
    with torch.no_grad():
        for byte in generated:
            input_tensor = torch.tensor([[byte]], dtype=torch.long)
            _, states = model(input_tensor, states=states, return_state=True)

        # Generate 30 new bytes autoregressively
        for _ in range(30):
            input_tensor = torch.tensor([[generated[-1]]], dtype=torch.long)
            logits, states = model(input_tensor, states=states, return_state=True)

            # Greedy decoding
            next_byte = torch.argmax(logits[0, 0], dim=-1).item()
            generated.append(next_byte)

            # Print character
            try:
                char = bytes([next_byte]).decode("utf-8")
                print(char, end="", flush=True)
            except UnicodeDecodeError:
                print(".", end="", flush=True)

    # Print final decoded string
    print("\n\nFull Generated text (decoded):")
    final_text = bytes(generated).decode("utf-8", errors="replace")
    print(f"'{final_text}'")
    print("--------------------------------------")


if __name__ == "__main__":
    main()
