# Visual Tracking — Goalkeeper Assistant

A Python visual tracking project for penalty-kick analysis, using MOG2 background subtraction + a Kalman filter to track the ball and predict its short-term trajectory from fixed-camera video.

## Contents
- `notebook.ipynb`: main notebook (pipeline + demo)
- `penalty_manual.csv`: manual clip timestamps
- `penalty_clips_manual/`: extracted penalty clips (generated locally)

## Notes
- Large media files are ignored by default via `.gitignore`.
- Input videos are expected locally (e.g. `input_h264.mp4`).
