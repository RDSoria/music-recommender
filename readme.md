# Audio Recommendation System

This project provides audio feature extraction and recommendation capabilities using two different approaches: a custom PoC extractor and an OpenL3-based extractor.

## Setup

### 1) Create virtual environment (Python 3.11+ recommended)
```bash
python -m venv .venv && source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install --upgrade pip
pip install numpy scipy joblib librosa soundfile scikit-learn pyloudnorm
pip install openl3 tensorflow-macos librosa soundfile numpy scikit-learn pyloudnorm
```

## Feature Extractors

### PoC Extractor (extractor_poc.py)
A proof-of-concept extractor that computes traditional audio features:
- **Features**: BPM, musical key, loudness (LUFS/RMS), 128-D embedding from spectral/MFCC/chroma features
- **Method**: Uses librosa for feature extraction, applies PCA for dimensionality reduction
- **Output**: One JSON file per audio file with extracted features

```bash
python extractor_poc.py --music-dir "/path/to/folder" --out-dir "./features_json" --seconds 75
```

**Parameters:**
- `--music-dir`: Directory containing audio files (mp3, flac, wav, m4a, ogg, aac)
- `--out-dir`: Output directory for JSON feature files
- `--seconds`: Analyze first N seconds (0 = full file)

### OpenL3 Extractor (extractor_openl3.py)
Advanced extractor using OpenL3 deep learning embeddings:
- **Features**: BPM, musical key, loudness, plus high-quality OpenL3 embeddings
- **Method**: Uses pre-trained OpenL3 models for rich audio representations
- **Output**: JSON files with OpenL3-based embeddings and traditional features
- **Advantages**: Better audio understanding, stereo-aware processing, configurable parameters

```bash
python extractor_openl3.py \
  --music-dir "/path/to/folder" \
  --out-dir "./features_json" \
  --seconds 0 --hop-size 0.1 --embedding-size 6144
```

**Parameters:**
- `--music-dir`: Directory containing audio files
- `--out-dir`: Output directory for JSON feature files
- `--seconds`: Analyze first N seconds (0 = full file)
- `--hop-size`: Time step for OpenL3 analysis (default: 0.1s)
- `--embedding-size`: OpenL3 embedding dimension (512, 6144)
- `--input-repr`: Input representation (mel128, mel256, linear)
- `--content-type`: Content type (music, env)
- `--reduce-to-128`: Reduce embeddings to 128-D using PCA

## Music Recommendation

### test_recommend.py
Recommendation engine that finds similar songs based on extracted features:
- **Scoring**: Combines embedding similarity (70%), harmonic compatibility (15%), BPM closeness (10%), and loudness similarity (5%)
- **Input**: Directory of JSON feature files from either extractor
- **Output**: Ranked list of similar songs with detailed scoring breakdown

```bash
python test_recommend.py --features-dir "./features_json" --song "Take on Me" --topk 3
```

**Parameters:**
- `--features-dir`: Directory containing JSON feature files
- `--song`: Seed song name (case-insensitive, partial matching supported)
- `--topk`: Number of recommendations to return

## Workflow

1. **Extract Features**: Use either extractor to process your music library
2. **Get Recommendations**: Use test_recommend.py to find similar songs
3. **Compare Results**: Try both extractors to see which works better for your music collection

## Supported Audio Formats
- MP3, FLAC, WAV, M4A, OGG, AAC
