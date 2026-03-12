# zn_o_coordination

This module tracks Zn-O coordination changes along AIMD trajectories.

## Goal

- Compute Zn coordination numbers frame by frame
- Detect adsorption, desorption, and coordination shifts
- Export coordination timelines and key event frames

## Expected inputs

- AIMD trajectory frames
- Zn-O cutoff or neighbor rule

## Expected outputs

- `outputs/coordination_table.csv`
- `outputs/coordination_events.csv`
- `outputs/key_frames/`

## Minimal workflow

1. Read frames
2. Count Zn-O neighbors
3. Compare coordination numbers over time
4. Save event tables and selected structures
