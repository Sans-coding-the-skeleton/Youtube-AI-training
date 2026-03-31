"""
model.py
ViralityNet — Multimodal architecture:
  - Vision branch: MobileNetV3-Small (frozen pretrained) → Linear projection
  - Text branch:   Embedding → GRU
  - Meta branch:   Numerical features + Category embedding → FC
  - Fusion:        Concat all → 2-layer FC → 1 output (log view count)
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ViralityNet(nn.Module):
    """
    Inputs:
      image       [B, 3, 224, 224]  - normalized thumbnail
      title_tokens [B, T]           - tokenized title (padded)
      num_features [B, 5]           - [duration, month, day_of_week, title_len, channel_follower_count] (normalized)
      cat_feature  [B]              - category index
      channel_feature [B]           - channel index

    Output:
      [B] - predicted log1p(view_count)
    """

    VISION_DIM = 576   # MobileNetV3-Small avgpool output channels (ends with 96→576 Conv1x1)
    VISION_PROJ = 256  # Projected vision dim
    TEXT_EMB_DIM = 64
    RNN_HIDDEN = 64
    CAT_EMB_DIM = 16
    CHAN_EMB_DIM = 16
    META_OUT = 32
    META_NUM_DIM = 5   # duration, month, day_of_week, title_len, channel_follower_count

    def __init__(self, num_cats: int, vocab_size: int, num_channels: int):
        super().__init__()

        # ── Vision Branch (frozen) ──────────────────────────────────────────
        backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )
        self.vision_features = backbone.features  # [B, 96, 7, 7]
        self.vision_pool = backbone.avgpool        # [B, 96, 1, 1]

        # Freeze all backbone weights — we use it purely as a feature extractor
        for param in self.vision_features.parameters():
            param.requires_grad = False
            
        # Unfreeze the last two blocks for fine-tuning on Youtube thumbnails
        for param in self.vision_features[-2:].parameters():
            param.requires_grad = True

        # Trainable projection head on top of features
        self.vision_proj = nn.Sequential(
            nn.Linear(self.VISION_DIM, self.VISION_PROJ),
            nn.BatchNorm1d(self.VISION_PROJ),
            nn.ReLU(),
            nn.Dropout(0.4),
        )

        # ── Text Branch ─────────────────────────────────────────────────────
        self.text_embedding = nn.Embedding(vocab_size + 1, self.TEXT_EMB_DIM, padding_idx=0)
        self.rnn = nn.GRU(
            self.TEXT_EMB_DIM, self.RNN_HIDDEN,
            batch_first=True, bidirectional=True
        )

        # ── Metadata Branch ─────────────────────────────────────────────────
        self.cat_embedding = nn.Embedding(num_cats + 1, self.CAT_EMB_DIM)
        self.channel_embedding = nn.Embedding(num_channels + 1, self.CHAN_EMB_DIM)
        self.meta_fc = nn.Sequential(
            nn.Linear(self.META_NUM_DIM + self.CAT_EMB_DIM + self.CHAN_EMB_DIM, self.META_OUT),
            nn.ReLU(),
        )

        # ── Fusion ──────────────────────────────────────────────────────────
        # Doubled RNN_HIDDEN due to bidirectional=True
        fusion_in = self.VISION_PROJ + (self.RNN_HIDDEN * 2) + self.META_OUT  
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, image, title_tokens, num_features, cat_feature, channel_feature):
        # Vision
        with torch.no_grad():
            v = self.vision_features(image)    # [B, 96, 7, 7]
            v = self.vision_pool(v)            # [B, 96, 1, 1]
        v = v.flatten(1)                       # [B, 96]
        v = self.vision_proj(v)                # [B, 256]

        # Text
        emb = self.text_embedding(title_tokens)   # [B, T, 64]
        _, h_n = self.rnn(emb)                    # h_n: [2, B, 64]
        # Concat last hidden state of forward and backward passes
        t = torch.cat([h_n[-2], h_n[-1]], dim=1)  # [B, 128]

        # Metadata
        cat_emb = self.cat_embedding(cat_feature)         # [B, 16]
        chan_emb = self.channel_embedding(channel_feature) # [B, 16]
        meta_in = torch.cat([num_features, cat_emb, chan_emb], 1)   # [B, 5+16+16=37]
        m = self.meta_fc(meta_in)                          # [B, 32]

        # Fuse
        combined = torch.cat([v, t, m], dim=1)   # [B, 352]
        out = self.fusion(combined)               # [B, 1]
        return out.squeeze(1)                     # [B]
