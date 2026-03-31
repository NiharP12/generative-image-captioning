"""
Inference and BLEU evaluation for Image Caption Generator.
"""
import os, sys
import torch
from PIL import Image
from nltk.translate.bleu_score import corpus_bleu

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.model import EncoderCNN, DecoderRNN
from app.utils import Vocabulary, load_captions
from app.dataset import get_transforms


def load_model(model_path, vocab_path, device):
    """Load trained encoder, decoder and vocabulary."""
    vocab = Vocabulary.load(vocab_path)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    encoder = EncoderCNN(embed_size=checkpoint["embed_size"]).to(device)
    decoder = DecoderRNN(
        embed_size=checkpoint["embed_size"],
        hidden_size=checkpoint["hidden_size"],
        vocab_size=checkpoint["vocab_size"],
    ).to(device)

    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    encoder.eval()
    decoder.eval()

    return encoder, decoder, vocab


def generate_caption(image_path, encoder, decoder, vocab, device, transform=None):
    """Generate a caption for a single image."""
    if transform is None:
        transform = get_transforms()

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = encoder(image_tensor)
        caption = decoder.generate(features, vocab)

    return caption


def evaluate_bleu(encoder, decoder, vocab, captions_dict, image_dir,
                  device, num_samples=None):
    """Compute corpus BLEU-1 through BLEU-4."""
    transform = get_transforms()
    references, hypotheses = [], []

    images = list(captions_dict.keys())
    if num_samples:
        images = images[:num_samples]

    for img_name in images:
        img_path = os.path.join(image_dir, img_name)
        if not os.path.exists(img_path):
            continue

        caption = generate_caption(img_path, encoder, decoder, vocab,
                                   device, transform)
        hypothesis = caption.split()
        refs = [ref.lower().split() for ref in captions_dict[img_name]]

        references.append(refs)
        hypotheses.append(hypothesis)

    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))

    return {"BLEU-1": bleu1, "BLEU-2": bleu2, "BLEU-3": bleu3, "BLEU-4": bleu4}


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pth")
    VOCAB_PATH = os.path.join(BASE_DIR, "models", "vocab.json")
    IMAGE_DIR = os.path.join(BASE_DIR, "Images")
    CAPTION_FILE = os.path.join(BASE_DIR, "captions.txt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading model...")
    encoder, decoder, vocab = load_model(MODEL_PATH, VOCAB_PATH, device)

    print("Evaluating BLEU scores on 200 images...")
    captions_dict = load_captions(CAPTION_FILE)
    scores = evaluate_bleu(encoder, decoder, vocab, captions_dict,
                           IMAGE_DIR, device, num_samples=200)

    print("\n" + "=" * 40)
    print("  BLEU Score Results")
    print("=" * 40)
    for metric, score in scores.items():
        print(f"  {metric}: {score:.4f}")
    print("=" * 40)
