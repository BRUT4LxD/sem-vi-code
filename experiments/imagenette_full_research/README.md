# ImageNette full research suite

End-to-end pipeline rooted at `final_research/` (override via `paths.root` in `config.yaml`).

## Prerequisites

- ImageNette layout: `./data/imagenette/train` and `./data/imagenette/val` (see `FullResearchPaths` in `paths.py`).
- GPU recommended; CPU will work but slowly.

## Run

From the repository root:

```bash
python -m experiments.imagenette_full_research.runner
```

`config.yaml` is the only input surface. Edit `run.resume`, `run.phases`, `training.architectures`, and `attacks.names` there before launching the runner.

Available phases for `run.phases`:

- `train_baseline`, `validate_normal`, `direct_normal`
- `progressive_active`, `validate_progressive_active`, `direct_progressive_active`
- `passive`, `validate_passive`, `direct_passive`
- `noise`, `transferability`

## Notes

- TensorBoard event files go under `final_research/runs/` (subfolders such as `imagenette_training/`, `adversarial_training/`, `binary_training/`, etc.).
- Baseline checkpoints use `{arch}_advanced.pt` under `final_research/models/normal/` (aligned with `ImageNetteTrainingConfigs.ADVANCED`).
- Saved adversarial PNGs use the checkpoint **stem** as the per-model folder name so file-based transferability can resolve `train/{stem}/{attack}/…`.
- Direct attack metrics on disk use saved **test** split images; perturbation distance columns are zero when only PNGs are available (no clean source stored).
