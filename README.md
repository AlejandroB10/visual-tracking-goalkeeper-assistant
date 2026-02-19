# visual-tracking-goalkeeper-assistant

Visual tracking project for penalty-kick analysis.

Current focus is a clean and reliable **initial ball detection** stage (MOG2-based) that can be handed off to a teammate for downstream tracking (template matching + optical flow).

## Current Notebook Status

Main notebook:

- `visual-tracking-goalkeeper.ipynb`

Helper module:

- `ball_detection_helpers.py`

Video:

- `penaltieshd.mp4`

Section 3 of the notebook has been simplified to:

1. Hardcoded penalty windows (no CSV reading, no dialogs)
2. Initial ball detection using MOG2 + lightweight filtering
3. Multi-penalty run summary (`p1` to `p4`)
4. Debug visualization with explicit ball markers

## Detection Scope

Implemented in notebook:

- OpenCV `MOG2` foreground subtraction
- Morphological cleanup (`open` + `close`)
- Candidate filtering by shape, color, and spatial priors
- Reliability checks across frames (`min_hits`, `max_jump`, score threshold)
- Output handoff format for teammate:

```python
init = {"frame": frame_i, "x": x, "y": y, "confidence": c}
```

Not implemented here on purpose:

- Template matching
- Optical flow tracking
- Goalkeeper direction prediction

Those are reserved for the teammate integration stage.

## Run Instructions

Open and run the notebook:

```bash
jupyter notebook visual-tracking-goalkeeper.ipynb
```

Run all cells in order, especially:

- `3.3 Demo run`
- `3.4 Debug visualization (all four penalties)`

## Current 3.3 Output (Reference)

Latest validated output from the notebook:

- `p1`: `frame=121, x=979, y=576, conf=0.770`
- `p2`: `frame=361, x=825, y=800, conf=0.633`
- `p3`: `frame=541, x=942, y=734, conf=0.802`
- `p4`: `frame=721, x=972, y=578, conf=0.790`

First valid initialization selected for handoff:

```python
{"frame": 121, "x": 979, "y": 576, "confidence": 0.77, "penalty": "p1"}
```
