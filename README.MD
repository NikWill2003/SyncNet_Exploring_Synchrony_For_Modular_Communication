# Exploring Synchrony for Modular Communication

Code for the MSc thesis. The SyncNet is in `src/models/syncnet/`, the baselines in `src/models/baselines/`

## Install

```bash
pip install -r requirements.txt
wandb login # optional
```

set `wandb.enabled=false` in the config to run without wandb logging

## Prepare the data

```bash
python -m scripts.prepare_dataset task=sort_of_clevr
python -m scripts.prepare_dataset task=sqoop dataset.rhs_variety=1
python -m scripts.prepare_dataset task=sqoop dataset.rhs_variety=2
python -m scripts.prepare_dataset task=sqoop dataset.rhs_variety=4
python -m scripts.prepare_dataset task=sqoop dataset.rhs_variety=8
python -m scripts.prepare_dataset task=sqoop dataset.rhs_variety=18
python -m scripts.prepare_dataset task=sqoop dataset.rhs_variety=35
```

## Sort-of-CLEVR

```bash
python -m scripts.main task=sort_of_clevr experiment=sort_of_clevr/syncnet
python -m scripts.main task=sort_of_clevr experiment=sort_of_clevr/baselines/question_only
python -m scripts.main task=sort_of_clevr experiment=sort_of_clevr/baselines/conv
python -m scripts.main task=sort_of_clevr experiment=sort_of_clevr/baselines/relnet
python -m scripts.main task=sort_of_clevr experiment=sort_of_clevr/baselines/film
python -m scripts.main task=sort_of_clevr experiment=sort_of_clevr/baselines/transformer
python -m scripts.main task=sort_of_clevr experiment=sort_of_clevr/baselines/shared_workspace

python -m scripts.main task=sort_of_clevr experiment=sort_of_clevr/syncnet_ablations
python -m scripts.main task=sort_of_clevr experiment=sort_of_clevr/syncnet_sweep_T
python -m scripts.main task=sort_of_clevr experiment=sort_of_clevr/syncnet_sweep_M
python -m scripts.main task=sort_of_clevr experiment=sort_of_clevr/syncnet_sweep_d
python -m scripts.main task=sort_of_clevr experiment=sort_of_clevr/syncnet_seeding
python -m scripts.main task=sort_of_clevr experiment=sort_of_clevr/screens/token_models
```

## SQOOP

```bash
python -m scripts.main task=sqoop experiment=sqoop/syncnet
python -m scripts.main task=sqoop experiment=sqoop/baselines/question_only
python -m scripts.main task=sqoop experiment=sqoop/baselines/conv
python -m scripts.main task=sqoop experiment=sqoop/baselines/relnet
python -m scripts.main task=sqoop experiment=sqoop/baselines/film
python -m scripts.main task=sqoop experiment=sqoop/baselines/transformer
python -m scripts.main task=sqoop experiment=sqoop/baselines/shared_workspace

python -m scripts.main task=sqoop experiment=sqoop/syncnet_ablations
python -m scripts.main task=sqoop experiment=sqoop/syncnet_sweep_T
python -m scripts.main task=sqoop experiment=sqoop/syncnet_sweep_M
python -m scripts.main task=sqoop experiment=sqoop/syncnet_sweep_d
python -m scripts.main task=sqoop experiment=sqoop/syncnet_seeding
python -m scripts.main task=sqoop experiment=sqoop/syncnet_grid
python -m scripts.main task=sqoop experiment=sqoop/screens/token_models
```

## Figures and tables
`figures.ipynb` pulls the runs from wandb and generate the figures and the code