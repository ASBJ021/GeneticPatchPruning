# Repository Structure

Overview of the main directories and key files :

```text
.
|-- README.md

|-- config/
|   |-- config.yaml
|   |-- data_colletion.yaml
|   |-- patch_selector_model.yaml
|   `-- training_config.yaml
|-- data/
|   `-- clane9/
|       |-- imagenet-100_10.jsonl
|       |-- imagenet-100_100_*.jsonl
|       |-- imagenet-100_1221_0.3_*.jsonl
|       |-- imagenet-100_1500_*.jsonl
|       `-- imagenet-100_500_*.jsonl
|-- gpp/
|   |-- __init__.py
|   |-- data_collection/
|   |   |-- __init__.py
|   |   `-- genetic_patch_pruning.py
|   |-- dataset/
|   |   |-- data_utils.py
|   |   |-- dataset.py
|   |   `-- dataset_patch_selection.py
|   |-- eval/
|   |   |-- compare_og_ga.py
|   |   `-- evaluate.py
|   |-- genetic_algo/
|   |   |-- __init__.py
|   |   |-- config.py
|   |   |-- ga.py
|   |   |-- io.py
|   |   |-- runner.py
|   |   |-- scoring.py
|   |   `-- stats.py
|   |-- model/
|   |   |-- clip_model.py
|   |   `-- model.py
|   |-- train/
|   |   `-- train.py
|   `-- utils/
|       `-- visual_utils.py
|-- main.py
|-- model_checkpoints/
|   `-- Test/
|       |-- checkpoint_best.pt
|       |-- checkpoint_epoch_001.pt
|       `-- checkpoint_epoch_002.pt

```
