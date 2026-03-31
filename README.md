# YouTube Virality Predictor

This modular AI package features an advanced multimodal neural network designed to estimate and predict the exact raw view count of a YouTube video based strictly on its compiled metadata, structural patterns, and high-resolution thumbnail characteristics. 

## 🧠 Model Architecture: `ViralityNet`
The prediction core uses a unified multimodal feature pipeline:
- **Vision Branch:** MobileNetV3-Small deep convolutional backbone partially fine-tuned for high-contrast YouTube thumbnails, scaled via 1D Batch Normalization.
- **Text Output:** A bidirectional GRU recurrent network digesting and scoring 64-dimension tokenized titles. 
- **Metadata Output:** Deep categorical matrix embeddings capturing explicit Channel Identity alongside structured numerical arrays handling normalized `channel_follower_count`, video duration, title length, and release date cycles.
- **Fusion Linear Head:** Heavy `Dropout(0.4)` dual-concatenated fusion layers predicting the standardized logarithmic scale.

## 🗃️ Dataset Compiler API
The background `yt-dlp` scraping scripts handle rate limitations manually. The dataset supports robust automated feature extractions, image caching (plus random `fallback` fillers via picsum APIs), and a fast `.json` to `.csv` preprocessor.

## Getting Started
To retrain or modify the model:
1. Run `python resume_dataset.py` to collect additional metrics until you hit your target size.
2. Run `python generate_csv.py` to bake all collected metadata files rapidly.
3. Access `thumbnail_virality_predictor/download_thumbnails.py` to fetch fresh image arrays via multithreading.
4. Run `python train.py` inside the predictive subspace to overwrite the `virality_model.pth` files.

### 📋 Ignored Objects
By default, the 2.0GB+ `dataset/` arrays, cached `thumbnails/` buffers, and `.pth` weight checkpoints natively circumvent source version controls (GitHub/Gitlab) to prevent upload failures. Use `git-lfs` specifically if you plan to share trained checkpoints!
