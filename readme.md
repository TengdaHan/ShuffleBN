# Shuffle BatchNorm

## Note: please refer to [github.com/facebookresearch/moco](https://github.com/facebookresearch/moco/blob/main/moco/builder.py#L69) for official implementation. This implementation is not verified.

An implementation of __Shuffle BatchNorm__ technique mentioned in [He et al., Momentum Contrast for Unsupervised Visual Representation Learning, 2019](https://arxiv.org/abs/1911.05722), in Section 3.3 "Shuffling BN". 

Implemented with torch 1.3.1. It works with pytorch [DistrbutedDataParallel](https://pytorch.org/docs/stable/nn.html?highlight=distributeddataparallel#torch.nn.parallel.DistributedDataParallel) with 1 process per GPU. So in order to use this `ShuffleBatchNorm` layer you need at least 2 GPUs. 

## What's this?

<img src="bn_algorithm.png" width="400"/>

The formula above is the BatchNorm algorithm. The `ShuffleBatchNorm` layer shuffles the mini-batch statistics (mean and variance) across multiple GPUs to avoid information leak. This operation eliminates model "cheating" when training contrastive loss and the contrast is obtained within the mini batch. 

## How to use?

The implementation mimics the design of [SyncBatchNorm](https://pytorch.org/docs/stable/nn.html?highlight=syncbatchnorm#torch.nn.SyncBatchNorm). To use `ShuffleBatchNorm`, just create your model first and then convert all `torch.nn.BatchNormND` layers into `ShuffleBatchNorm` by the function:
  ```python
  from shuffle_batchnorm import ShuffleBatchNorm
  # ...
  model = Model() # with BN layers
  model = ShuffleBatchNorm.convert_shuffle_batchnorm(model)
  ```
See `main.py` for a completed example. 

## Check result
run command:
```bash
$ python main.py --gpu 0,1 --shuffle --epochs 10
=> Spawning 2 distributed workers
...
[0]mean before shuffle: tensor([-0.2478,  0.1704,  0.0640, -0.2732], device='cuda:0')
[1]mean before shuffle: tensor([-0.4012, -0.1913, -0.0553, -0.1917], device='cuda:1')
[0]mean after shuffle: tensor([-0.4012, -0.1913, -0.0553, -0.1917], device='cuda:0')
[1]mean after shuffle: tensor([-0.2478,  0.1704,  0.0640, -0.2732], device='cuda:1')
[9/10] Loss 0.6868
================================================
[9/10] Loss 0.7908
================================================
```

## Notes
If you find bugs, please create an issue. Very welcome!

Update:
* Doesn't work when training with multiple nodes, will fix soon.
