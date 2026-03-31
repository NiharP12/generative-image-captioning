"""
Training script for Image Caption Generator.
Uses ResNet50 encoder (Phase 2) + LSTM decoder.
"""
import os, sys, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.model import EncoderCNN, DecoderRNN
from app.dataset import FlickrDataset, get_transforms, CaptionCollate
from app.utils import Vocabulary, load_captions, get_all_captions


def train():
    # ── Paths ──
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    IMAGE_DIR = os.path.join(BASE_DIR, "Images")
    CAPTION_FILE = os.path.join(BASE_DIR, "captions.txt")
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Hyperparameters ──
    EMBED_SIZE = 256
    HIDDEN_SIZE = 512
    NUM_LAYERS = 1
    DROPOUT = 0.5
    BATCH_SIZE = 16
    NUM_EPOCHS = 20
    ENCODER_LR = 1e-5
    DECODER_LR = 1e-4
    FREQ_THRESHOLD = 5

    # ── Device ──
    device = torch.device("cpu")
    print(f"{'='*60}")
    print(f"  Image Caption Generator - Training")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # ── Load & Clean Captions ──
    print("[1/6] Loading captions...")
    captions_dict = load_captions(CAPTION_FILE)
    all_captions = get_all_captions(captions_dict)
    print(f"  Images: {len(captions_dict)} | Captions: {len(all_captions)}")

    # ── Build Vocabulary ──
    print("[2/6] Building vocabulary...")
    vocab = Vocabulary(freq_threshold=FREQ_THRESHOLD)
    vocab.build_vocabulary(all_captions)
    vocab_size = len(vocab)
    print(f"  Vocabulary size: {vocab_size}")
    vocab.save(os.path.join(MODEL_DIR, "vocab.json"))

    # ── Dataset & DataLoader ──
    print("[3/6] Preparing dataset...")
    transform = get_transforms()
    dataset = FlickrDataset(IMAGE_DIR, captions_dict, vocab, transform)
    print(f"  Dataset size: {len(dataset)}")

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                     generator=torch.Generator().manual_seed(42))

    pad_idx = vocab.word2idx["<pad>"]
    collate_fn = CaptionCollate(pad_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, collate_fn=collate_fn, pin_memory=True)

    # ── Models ──
    print("[4/6] Building models...")
    encoder = EncoderCNN(embed_size=EMBED_SIZE).to(device)
    decoder = DecoderRNN(embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE,
                         vocab_size=vocab_size, num_layers=NUM_LAYERS,
                         dropout=DROPOUT).to(device)

    trainable_enc = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    trainable_dec = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"  Encoder trainable params: {trainable_enc:,}")
    print(f"  Decoder trainable params: {trainable_dec:,}")

    # ── Loss & Optimizer ──
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam([
        {"params": encoder.parameters(), "lr": ENCODER_LR},
        {"params": decoder.parameters(), "lr": DECODER_LR}
    ], foreach=False)


    # ── Training Loop ──
    print(f"\n[5/6] Training for {NUM_EPOCHS} epochs...\n")
    best_val_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        t0 = time.time()

        # --- Train ---
        encoder.train()
        decoder.train()
        train_loss = 0.0

        for batch_idx, (images, captions) in enumerate(train_loader):
            images = images.to(device)
            captions = captions.to(device)

            optimizer.zero_grad(set_to_none=True)

            features = encoder(images)
            outputs = decoder(features, captions)

            # outputs: (B, L, V), targets: full captions (B, L)
            loss = criterion(outputs.reshape(-1, outputs.size(-1)),
                             captions.reshape(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f}")

        avg_train = train_loss / len(train_loader)

        # --- Validate ---
        encoder.eval()
        decoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, captions in val_loader:
                images = images.to(device)
                captions = captions.to(device)
                features = encoder(images)
                outputs = decoder(features, captions)
                loss = criterion(outputs.reshape(-1, outputs.size(-1)),
                                 captions.reshape(-1))
                val_loss += loss.item()
        avg_val = val_loss / len(val_loader)

        elapsed = time.time() - t0
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
              f"Train: {avg_train:.4f} | Val: {avg_val:.4f} | "
              f"Time: {elapsed:.1f}s")

        # Save best
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "epoch": epoch,
                "vocab_size": vocab_size,
                "embed_size": EMBED_SIZE,
                "hidden_size": HIDDEN_SIZE,
                "val_loss": best_val_loss,
            }, os.path.join(MODEL_DIR, "model.pth"))
            print(f"  >>> Model saved (val_loss={best_val_loss:.4f})")

    print(f"\n[6/6] Training complete! Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train()
