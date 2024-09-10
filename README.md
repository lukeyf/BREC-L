# Theoretical Insights into Line Graph Transformation on Graph Learning

## About

This repository is the implementation of the following paper: [Theoretical Insights into Line Graph Transformation on Graph Learning]().


This project is built on the [**BREC** dataset](https://github.com/GraphPKU/BREC) which includes 400 pairs of graphs categorized into basic, regular, extension, and CFI graphs. The following dictionary shows the indices of these graphs in the 400 pairs.

```python
  "Basic": (0, 60),
  "Regular": (60, 110),
  "Strongly Regular": (110, 160),
  "Extension": (160, 260),
  "CFI": (260, 360),
  "4-Vertex_Condition": (360, 380),
  "Distance_Regular": (380, 400),
```


## Usages

### File Structure

We first introduce the general file structure of BREC:

```bash
├── Data
    └── raw
        └── brec_v3.npy    # unprocessed BREC dataset in graph6 format
├── BRECDataset_v3.py    # BREC dataset construction file
├── test_BREC.py    # Evaluation framework file
└── test_BREC_search.py    # Run test_BREC.py with 10 seeds for the final result
```

To test on BREC, there are four steps to follow:

1. Select a model and go to the corresponding [directory](#directory).
2. [Prepare](#preparation) dataset based on selected model requirements.
3. Check *test_BREC.py* for implementation if you want to test your own GNN.
4. Run *test_BREC_search.py* for final result. Only if no failure in reliability check for all seeds is available.

### Requirements

The experiments were run on: Python 3.8.13 + [PyTorch 1.13.1](https://pytorch.org/get-started/previous-versions/) + [PyTorch_Geometric 2.2](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
You can use the following command to build the torch environment.


```bash
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
pip install torch-geometric==2.2.0
```
You can use the following command for the rest of the environment building.

```bash
pip install -r requirements.txt
```


### <span id="preparation">Data Preparation</span>

Data preparation requires two steps: generate the dataset and arrange it in the correct position.


First, unzip the dataset by

```
unzip BREC_data_all.zip
```

Only the PPGN involves the usage of `brec_v3.npy`. Move this file to `ProvablyPowerfulGraphNetworks_torch/Data/raw/`.


### <span id="reproduce">Reproducing PPGN</span>

First, move to the directory using 

```bash
cd ProvablyPowerfulGraphNetworks_torch/main_scripts
```

For the non-line graph experiment, you can use 
```bash
python test_BREC_search.py
```

For the line graph experiment, you can use

```bash
python test_BREC_search_line.py
```

### <span id="reproduce">Reproducing WL Tests</span>

First, move to the directory using 
```
cd Non-GNNs
```

To reproduce result on 3-WL, run:

```bash
python test.py --wl 2 --method fwl
```

or

```bash
python test.py --wl 3 --method k-wl
```

For 4-WL, you can use 

```bash
python test.py --wl 3 --method fwl
```
or

```bash
python test.py --wl 4 --method k-wl
```

For line graph experiment for example with the 3-WL test, use 

```bash
python test.py --wl 2 --method fwl --line_graph_degree 1
```