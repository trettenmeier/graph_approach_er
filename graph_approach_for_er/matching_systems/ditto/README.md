python train_ditto.py --task mp_dataset_medium --batch_size 64 --max_len 256 --lr 3e-5 --n_epochs 40 --lm distilbert --summarize --finetuning




Modified Ditto-Repository (cloned on 20.6.2023)

Python environment 3.7.7, cuda 11.3, pip freeze:

absl-py==1.4.0
anyio==3.7.0
apex @ file:///D:/bld/nvidia-apex_1619107510166/work
argon2-cffi==21.3.0
argon2-cffi-bindings==21.2.0
attrs==23.1.0
backcall==0.2.0
beautifulsoup4==4.12.2
bleach==6.0.0
blis==0.7.9
cachetools==5.3.1
catalogue==2.0.8
certifi==2023.5.7
cffi==1.15.1
charset-normalizer==3.1.0
click==7.1.2
colorama @ file:///home/conda/feedstock_root/build_artifacts/colorama_1666700638685/work
cymem==2.0.7
debugpy==1.6.7
decorator==5.1.1
defusedxml==0.7.1
en-core-web-lg @ https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.1.0/en_core_web_lg-3.1.0-py3-none-any.whl
entrypoints==0.4
exceptiongroup @ file:///home/conda/feedstock_root/build_artifacts/exceptiongroup_1678703645500/work
fastjsonschema==2.17.1
filelock==3.12.2
gensim==3.8.1
google-auth==2.20.0
google-auth-oauthlib==0.4.6
grpcio==1.54.2
huggingface-hub==0.0.12
idna==3.4
importlib-metadata==6.7.0
importlib-resources==5.12.0
iniconfig @ file:///home/conda/feedstock_root/build_artifacts/iniconfig_1673103042956/work
ipykernel==6.16.2
ipython==7.34.0
ipython-genutils==0.2.0
ipywidgets==8.0.6
jedi==0.18.2
Jinja2==3.1.2
joblib==1.2.0
jsonlines==1.2.0
jsonschema==4.17.3
jupyter==1.0.0
jupyter-console==6.6.3
jupyter-server==1.24.0
jupyter_client==7.4.9
jupyter_core==4.12.0
jupyterlab-pygments==0.2.2
jupyterlab-widgets==3.0.7
Markdown==3.4.3
MarkupSafe==2.1.3
matplotlib-inline==0.1.6
mistune==2.0.5
murmurhash==1.0.9
nbclassic==1.0.0
nbclient==0.7.4
nbconvert==7.5.0
nbformat==5.8.0
nest-asyncio==1.5.6
nltk==3.5
notebook==6.5.4
notebook_shim==0.2.3
numpy==1.19.2
oauthlib==3.2.2
packaging @ file:///home/conda/feedstock_root/build_artifacts/packaging_1681337016113/work
pandocfilters==1.5.0
parso==0.8.3
pathy==0.10.1
pickleshare==0.7.5
Pillow==9.3.0
pkgutil_resolve_name==1.3.10
pluggy @ file:///D:/bld/pluggy_1648772784311/work
preshed==3.0.8
prometheus-client==0.17.0
prompt-toolkit==3.0.38
protobuf==4.23.3
psutil==5.9.5
pyasn1==0.5.0
pyasn1-modules==0.3.0
pycparser==2.21
pydantic==1.8.2
Pygments==2.15.1
pyrsistent==0.19.3
pytest @ file:///home/conda/feedstock_root/build_artifacts/pytest_1686924711497/work
python-dateutil==2.8.2
pywin32==306
pywinpty==2.0.10
PyYAML @ file:///D:/bld/pyyaml_1648757357341/work
pyzmq==25.1.0
qtconsole==5.4.3
QtPy==2.3.1
regex==2019.12.20
requests==2.31.0
requests-oauthlib==1.3.1
rsa==4.9
sacremoses==0.0.53
scikit-learn==1.0.2
scipy==1.3.2
Send2Trash==1.8.2
sentencepiece==0.1.85
six==1.16.0
sklearn==0.0
smart-open==6.3.0
sniffio==1.3.0
soupsieve==2.4.1
spacy==3.1.0
spacy-legacy==3.0.12
srsly==2.4.6
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.1
tensorboardX==2.6.1
terminado==0.17.1
thinc==8.0.17
threadpoolctl==3.1.0
timm==0.4.12
tinycss2==1.2.1
tokenizers==0.10.3
tomli @ file:///home/conda/feedstock_root/build_artifacts/tomli_1644342247877/work
torch==1.9.0+cu111
torchaudio==0.9.0
torchvision==0.10.0
tornado==6.2
tqdm @ file:///home/conda/feedstock_root/build_artifacts/tqdm_1677948868469/work
traitlets==5.9.0
transformers==4.9.2
typer==0.3.2
typing-extensions==3.10.0.2
urllib3==1.25.4
wasabi==0.10.1
wcwidth==0.2.6
webencodings==0.5.1
websocket-client==1.6.0
Werkzeug==2.2.3
widgetsnbextension==4.0.7
wincertstore==0.2
zipp @ file:///home/conda/feedstock_root/build_artifacts/zipp_1677313463193/work






# Ditto: Deep Entity Matching with Pre-Trained Language Models

*Update: a new light-weight version based on new versions of Transformers*

Ditto is an entity matching (EM) solution based on pre-trained language models such as BERT. Given a pair of data entries, EM checks if the two entries refer to the same real-world entities (products, businesses, publications, persons, etc.). Ditto leverages the powerful language understanding capability of pre-trained language models (LMs) via fine-tuning. Ditto serializes each data entry into a text sequence and casts EM as a sequence-pair classification problem solvable by LM fine-tuning. We also employ a set of novel optimizations including summarization, injecting domain-specific knowledge, and data augmentation to further boost the performance of the matching models.

For more technical details, see the [Deep Entity Matching with Pre-Trained Language Models](https://arxiv.org/abs/2004.00584) paper.

## Requirements

* Python 3.7.7
* PyTorch 1.9
* HuggingFace Transformers 4.9.2
* Spacy with the ``en_core_web_lg`` models
* NVIDIA Apex (fp16 training)

Install required packages
```
conda install -c conda-forge nvidia-apex
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

## The EM pipeline

A typical EM pipeline consists of two phases: blocking and matching. 
![The EM pipeline of Ditto.](ditto.jpg)
The blocking phase typically consists of simple heuristics that reduce the number of candidate pairs to perform the pairwise comparisons. Ditto optimizes the matching phase which performs the actual pairwise comparisons. The input to Ditto consists of a set of labeled candidate data entry pairs. Each data entry is pre-serialized into the following format:
```
COL title VAL microsoft visio standard 2007 version upgrade COL manufacturer VAL microsoft COL price VAL 129.95
```
where ``COL`` and ``VAL`` are special tokens to indicate the starts of attribute names and attribute values. A complete example pair is of the format
```
<entry_1> \t <entry_2> \t <label>
```
where the two entries are serialized and ``<label>`` is either ``0`` (no-match) or ``1`` (match). In our experiments, we evaluated Ditto using two benchmarks:
* the [ER_Magellan benchmarks](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md) used in the [DeepMatcher paper](http://pages.cs.wisc.edu/~anhai/papers1/deepmatcher-sigmod18.pdf). This benchmark contains 13 datasets of 3 categories: ``Structured``, ``Dirty``, and ``Textual`` representing different dataset characteristics. 
* the [WDC product matching benchmark](http://webdatacommons.org/largescaleproductcorpus/v2/index.html). This benchmark contains e-commerce product offering pairs from 4 domains: ``cameras``, ``computers``, ``shoes``, and ``watches``. The training data of each domain is also sub-sampled into different sizes, ``small``, ``medium``, ``large``, and ``xlarge`` to test the label efficiency of the models. 

We provide the serialized version of their datasets in ``data/``. The dataset configurations can be found in ``configs.json``. 

## Training with Ditto

To train the matching model with Ditto:
```
CUDA_VISIBLE_DEVICES=0 python train_ditto.py \
  --task Structured/Beer \
  --batch_size 64 \
  --max_len 64 \
  --lr 3e-5 \
  --n_epochs 40 \
  --lm distilbert \
  --fp16 \
  --da del \
  --dk product \
  --summarize
```
The meaning of the flags:
* ``--task``: the name of the tasks (see ``configs.json``)
* ``--batch_size``, ``--max_len``, ``--lr``, ``--n_epochs``: the batch size, max sequence length, learning rate, and the number of epochs
* ``--lm``: the language model. We now support ``bert``, ``distilbert``, and ``albert`` (``distilbert`` by default).
* ``--fp16``: whether train with the half-precision floating point optimization
* ``--da``, ``--dk``, ``--summarize``: the 3 optimizations of Ditto. See the followings for details.
* ``--save_model``: if this flag is on, then save the checkpoint to ``{logdir}/{task}/model.pt``.

### Data augmentation (DA)

If the ``--da`` flag is set, then ditto will train the matching model with MixDA, a data augmentation technique for text data. To use data augmentation, one transformation operator needs to be specified. We currently support the following operators for EM:


| Operators       | Details                                           |
|-----------------|---------------------------------------------------|
|del              | Delete a span of tokens                      |
|swap             | Shuffle a span of tokens                          |
|drop_col         | Delete a whole attribute                          |
|append_col       | Move an attribute (append to the end of another attr) |
|all              | Apply all the operators uniformly at random    |

### Domain Knowledge (DK)

Inject domain knowledge to the input sequences if the ``--dk`` flag is set. Ditto will preprocess the serialized entries by
* tagging informative spans (e.g., product ID, persons name) by inserting special tokens (e.g., ID, PERSON)
* normalizing certain spans (e.g., numbers)
We currently support two injection modes: ``--dk general`` and ``--dk product`` for the general domain and for the product domain respectively. See ``ditto/knowledge.py`` for more details.

### Summarization
When the ``--summarize`` flag is set, the input sequence will be summarized by retaining only the high TF-IDF tokens. The resulting sequence will be of length no more than the max sequence length (i.e., ``--max_len``). See ``ditto/summarize.py`` for more details.

## To run the matching models
Use the command:
```
CUDA_VISIBLE_DEVICES=0 python matcher.py \
  --task wdc_all_small \
  --input_path input/input_small.jsonl \
  --output_path output/output_small.jsonl \
  --lm distilbert \
  --max_len 64 \
  --use_gpu \
  --fp16 \
  --checkpoint_path checkpoints/
```
where ``--task`` is the task name, ``--input_path`` is the input file of the candidate pairs in the jsonlines format, ``--output_path`` is the output path, and ``checkpoint_path`` is the path to the model checkpoint (same as ``--logdir`` when training). The language model ``--lm`` and ``--max_len`` should be set to the same as the one used in training. The same ``--dk`` and ``--summarize`` flags also need to be specified if they are used at the training time.

## Colab notebook

You can also run training and prediction using this colab [notebook](https://colab.research.google.com/drive/1eyQbockBSxxQ_tuW5F1XKyeVOM1HT_Ro?usp=sharing).
