# Graph-Based Data Augmentation and Label Noise Detection for Entity Resolution

This repository is the corresponding code for the paper `Graph-Based Data Augmentation and Label Noise Detection for Entity Resolution`.
The most relevant files as described in the paper are `trainer/bert_trainer.py` and `graph_augmentation/graph_augmentation.py`.

# Installation and setup:
- Python 3.11
- Torch 2.3
- Clone repository and `pip install -e .`
- Create `config.yml` in package-root with the entry `working_dir: '/path/to_some_directory'`
- Data is expected to be in this `working_dir/data/raw` (you can always look up the correct path in the .yml files in the experiments folder)

# Running experiments
Experiments are run via luigi-tasks:
- `python run_experiment.py` for a single experiment (specify these in `run_experiment.py`)
- `python run_all_experiments.py` for all experiments (this will take a very long time). 
   In the default settings the system luigi-scheduler (luigid) is being used, but you can 
   always change this in the file to `local_scheduler=False`
