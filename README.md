# Just-in-time Vulnerability Detection - Replication
## Generate data
* Dump pickle with protocol version 4.
* Run local or kaggle:
  * https://www.kaggle.com/code/ducanger/jit-vul-cc2vec-generate-data

  * https://www.kaggle.com/code/ducanger/jit-vul-la-generate-data

  * https://www.kaggle.com/code/ducanger/jit-vul-jitfine-generate-data

## Model
### VCCFinder
```
python -m baselines.VCCFinder.vccfinder -split by_time
```
```
python -m baselines.VCCFinder.vccfinder -split cross_project
```

### LAPredict
```
python -m baselines.LApredict.lapredict -split by_time
```
```
python -m baselines.LApredict.lapredict -split cross_project
```

### JITLine
k = 12, time = 2250s
Precision: 0.6136, Recall: 0.6626, F1: 0.6371,  AUC: 0.7716, ACC: 0.8008, PCI@20%LOC: 0.7408, Effort@20%Recall: 0.0161, POpt: 0.9384
```
python -m baselines.JITLine.jitline -split by_time
```
k = 17, time = 1031s

Precision: 0.5717, Recall: 0.7304, F1: 0.6414,  AUC: 0.6956, ACC: 0.7709, PCI@20%LOC: 0.7589, Effort@20%Recall: 0.0114, POpt: 0.9418
```
python -m baselines.JITLine.jitline -split cross_project
```
* Localization
```
python -m baselines.JITLine.jitline_localization -split by_time
```
```
python -m baselines.JITLine.jitline_localization -split cross_project
```

### JITFine
* Run on Kaggle: https://www.kaggle.com/code/ducanger/jit-vul-jitfine-replication