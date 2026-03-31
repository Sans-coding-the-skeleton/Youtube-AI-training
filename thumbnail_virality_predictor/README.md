# YouTube Thumbnail Virality Predictor

This multimodal neural network predicts the virality potential (log view count) of a YouTube video based on its thumbnail image, title text, category, and metadata.

## How to run the Web App (Teacher Demo)

Since the model (`virality_model.pth`) is already fully trained and saved, you **do not** need to re-run the training script.

To launch the web interface:
1. Open a terminal in this project folder (`thumbnail_virality_predictor`).
2. Run the Flask server using your Python environment:
   ```cmd
   C:\Python314\python.exe server.py
   ```
3. Open your web browser and go to `http://localhost:5000`
4. Paste any YouTube URL into the search bar to see its virality breakdown!

---

## Technical Architecture
* **Vision Branch**: Pre-trained `MobileNetV3-Small` (processes the thumbnail image).
* **Text Branch**: `GRU` architecture (processes the tokenized title text).
* **Metadata Branch**: Encodes categorical metadata like duration and upload date.
All three branches are fused to produce the final virality score prediction, alongside a dataset-relative percentile rank.
