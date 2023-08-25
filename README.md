# Just-in-time Vulnerability Detection - Replication
## Generate data
* Dump pickle with protocol version 4.
* Run local or kaggle:
  * https://www.kaggle.com/code/ducanger/jit-vul-cc2vec-generate-data

  * https://www.kaggle.com/code/ducanger/jit-vul-la-generate-data

  * https://www.kaggle.com/code/ducanger/jit-vul-jitfine-generate-data

## Model
### VCCFinder
    $ python -m baselines.VCCFinder.vccfinder -split by_time
    $ python -m baselines.VCCFinder.vccfinder -split cross_project

### LAPredict
    $ python -m baselines.LApredict.lapredict -split by_time
    $ python -m baselines.LApredict.lapredict -split cross_project

### JITLine
    $ python -m baselines.JITLine.jitline -split by_time
    $ python -m baselines.JITLine.jitline -split cross_project

### JITFine
* Run on Kaggle: https://www.kaggle.com/code/ducanger/jit-vul-jitfine-replication