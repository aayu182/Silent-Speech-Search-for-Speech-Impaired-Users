# Silent Speech Search for Speech-Impaired Users

A search interface for people who cannot speak. Instead of a microphone, the user silently mouths a word in front of a webcam; the system reads the lip movements and turns them into a search query, like voice search but without any sound.

> **Status:** work in progress. The data-collection tool is done. Model training and the search UI are in development.

## How it works

1. The webcam captures the user's face.
2. MediaPipe Face Landmarker locates the mouth region in every frame.
3. Only the cropped mouth area (96x96) is shown on screen and saved, never the full face.
4. A lip-reading model (CNN + LSTM) predicts the word from the sequence of mouth frames.
5. The predicted word appears in a search bar and the search runs.

## Privacy

- Everything runs locally on the user's computer.
- The full face is never displayed or stored; only the mouth crop is.
- Recorded videos stay out of this repository (`data/` is git-ignored).

## Roadmap

- [x] Real-time mouth-region detection and cropping
- [x] Data recording tool (`record_mouth.py`)
- [ ] Record dataset (30-50 clips per word, 5 words)
- [ ] Train CNN + LSTM word classifier
- [ ] Streamlit search UI with a mic-style button
- [ ] Add more words and languages (e.g. Hindi)

## Setup

Requires Python 3.10+.

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

## Record data

```bash
python record_mouth.py
```

| Key | Action |
|---|---|
| `1`-`5` | Choose word (weather, news, music, maps, hello) |
| `SPACE` | Record a 2-second clip (mouth the word silently) |
| `q` | Quit |

On first run, the script downloads the MediaPipe face model (`face_landmarker.task`, about 4 MB). Clips are saved to `data/<word>/`.

## Tech stack

Python, OpenCV, MediaPipe, NumPy, TensorFlow/PyTorch (planned), Streamlit (planned)