"""
Encoder-Decoder Model for Image Captioning.
- Encoder: ResNet50 (Phase 2 fine-tuning - only layer4 unfrozen)
- Decoder: LSTM with embedding and dropout
"""
import torch
import torch.nn as nn
from torchvision import models


class EncoderCNN(nn.Module):
    """ResNet50 encoder with Phase 2 partial fine-tuning."""

    def __init__(self, embed_size=256):
        super(EncoderCNN, self).__init__()

        # Load pretrained ResNet50
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Remove final FC layer (keep avgpool)
        # Children: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        # Phase 2: Freeze ALL layers first
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Unfreeze only layer4 (index 7 in Sequential)
        for param in self.resnet[7].parameters():
            param.requires_grad = True

        # Projection: 2048 -> embed_size
        self.fc = nn.Linear(2048, embed_size)
        self.bn = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        features = self.resnet(images)          # (B, 2048, 1, 1)
        features = features.view(features.size(0), -1)  # (B, 2048)
        features = self.dropout(self.bn(self.fc(features)))
        return features


class DecoderRNN(nn.Module):
    """LSTM decoder for caption generation."""

    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout=0.5):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, captions):
        """
        Teacher forcing training.
        features: (B, embed_size)
        captions: (B, L) padded caption indices [<start>, w1, ..., wn, <end>, <pad>...]
        Returns: (B, L, vocab_size)
        """
        # Embed all caption tokens except the last one
        embeddings = self.dropout(self.embed(captions[:, :-1]))  # (B, L-1, E)

        # Prepend image features as first input token
        features = features.unsqueeze(1)  # (B, 1, E)
        lstm_input = torch.cat([features, embeddings], dim=1)  # (B, L, E)

        lstm_out, _ = self.lstm(lstm_input)  # (B, L, H)
        outputs = self.fc(lstm_out)          # (B, L, V)
        return outputs

    def generate(self, features, vocab, max_length=50):
        """Greedy decoding for caption generation."""
        result = []
        x = features.unsqueeze(1)  # (1, 1, E)
        hidden = None

        with torch.no_grad():
            for _ in range(max_length):
                lstm_out, hidden = self.lstm(x, hidden)
                output = self.fc(lstm_out.squeeze(1))  # (1, V)
                predicted = output.argmax(dim=1)        # (1,)

                word_idx = predicted.item()
                word = vocab.idx2word[word_idx]
                if word == "<end>":
                    break
                if word != "<start>":
                    result.append(word)

                x = self.embed(predicted).unsqueeze(1)  # (1, 1, E)

        return " ".join(result)
