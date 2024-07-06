# MStar
This repository is the official implementation of MStar, the method proposed in paper "Expanding the Scope: Inductive Knowledge Graph Reasoning with Multi-Starting Progressive Propagation".


## Requirements
python=3.7

```
conda create -n MStar python=3.7
conda activate MStar
conda install pytorch==1.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install -r requirements.txt
pip install torch-scatter==2.0.8 -f https://data.pyg.org/whl/torch-1.8.0%2Bcu111.html
pip install torch-sparse==0.6.12 -f https://data.pyg.org/whl/torch-1.8.0%2Bcu111.html
```


## Reproduction

- [-D]  Dataset.
- [-T]  Task, i.e., train/test.
- [-HW] Employ highway layer if selecting HW.
- [-E]  Experiment name.


### Train MStar
```
python3 train.py -D fb237_v1 -T train -HW --gpu 0 -E reproduction
```

### Test MStar
```
python3 train.py -D fb237_v1 -T test -HW --gpu 0 -E reproduction
```

## Ablation
- [-M] Selection method. "None" removes entities selection. It works when not selecting HW.
- [--train_bad] Do not filter noisy samples if selecting train_bad.

```
python3 train.py -D fb237_v1 -T train --gpu 0 -M None -E wo_Selection
python3 train.py -D fb237_v1 -T train --gpu 0 -E wo_HighwayLayer
python3 train.py -D fb237_v1 -T train --gpu 0 -HW --train_bad -E wo_LinkVerify
```



## Per-distance Performance
Generate distance information for dataset `fb237_v1`
```
python3 analysis/dist_process.py -D fb237_v1
```
The distance information of `fb237_v1` is output to `analysis/dist_logs/dist_fb237_v1.log`.

The dataset `fb237_v1` with distance for per-distance performance testing is output to `data/fb237_v1_ind/test4.txt`.
Check per-distance performance by test and the result is output to `test_results.txt`.


## Acknowledgement
MStar is designed upon knowledge graph reasoning model [RED-GNN](https://github.com/LARS-research/RED-GNN). We thank them for making the code open-sourced.

## Citation

```
@inproceedings{MStar,
  title     = {Expanding the Scope: Inductive Knowledge Graph Reasoning with Multi-Starting Progressive Propagation},
  author    = {Shao, Zhoutian and 
               Cui, Yuanning and 
               Hu, Wei},
  booktitle = {ISWC},
  year      = {2024}
}
```