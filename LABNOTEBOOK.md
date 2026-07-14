# Lab Notebook

## Python 3.12 and Blackwell compatibility

This branch updates CPA for a modern Python and GPU software stack so it can train with torch CUDA 12.8 wheels on Blackwell GPUs.

Key implementation changes:

- Dependency bounds in `pyproject.toml` now allow Python 3.12, modern torch, modern scvi-tools, Lightning 2, current anndata, and the maintained `rdkit` package.
- GPU selection no longer depends on removed scvi helpers; CPA maps the old `use_gpu` argument to Lightning `accelerator` and `devices`.
- CPA imports Lightning callbacks from `lightning.pytorch` to avoid mixed `pytorch_lightning`/`lightning.pytorch` Trainer errors.
- `CPATrainingPlan` uses Lightning 2 epoch hooks (`on_train_epoch_end`, `on_validation_epoch_end`) and stores step outputs internally.
- AnnData splitting tolerates scvi versions without `settings.dl_pin_memory_gpu_training`, defaulting to pinned memory for GPU training.

Validation:

- `python -m compileall cpa` passes in the Python 3.12 CPA environment.
- A synthetic CPA one-epoch smoke test completed on an RTX PRO 6000 Blackwell GPU with torch `2.11.0+cu128`.
