# vdw_energy_predictor

This module predicts or estimates vdW-related energy contributions for candidate structures.

## Goal

- Score frames with a vdW surrogate or correction model
- Rank structures by vdW importance
- Export structures that should be retained in the training pool

## Expected inputs

- AIMD frames or selected candidate frames
- Predictor weights or reference data

## Expected outputs

- `outputs/vdw_scores.csv`
- `outputs/high_vdw_frames/`
- `outputs/summary.json`

## Minimal workflow

1. Load predictor
2. Evaluate candidate frames
3. Rank vdW-sensitive structures
4. Save selected frames
