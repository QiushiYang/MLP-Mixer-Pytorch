# MLP-Mixer-Pytorch
Unofficial PyTorch implementation of MLP-Mixer with the function of loading official ImageNet pre-trained parameters.

# Usage
```python
from mlp_mixer import MlpMixer

pretrain_model='./pretrain_models/imagenet21k_Mixer-B_16.npz'

model = MlpMixer(num_classes=10, 
                 num_blocks=4, 
                 patch_size=16, 
                 hidden_dim=768, 
                 tokens_mlp_dim=384, 
                 channels_mlp_dim=3072, 
                 image_size=224
                 )

# load official ImageNet pre-trained model:
model.load_from(np.load(pretrain_model))
print ('Finish loading the pre-trained model!')

num_param = sum(p.numel() for p in model.parameters()) / 1e6
print('Total params.: %f M'%num_param)

pred = model(img)
```

# Fine-tuning
Download the official pre-trained models at <https://console.cloud.google.com/storage/mixer_models/>. 

Hypyer-parameters setting for better fine-tuning:
```python
optim = torch.optim.SGD(param_list, 
                        lr=1e-3, 
                        weight_decay=0.0,
                        momentum=0.9, 
                        nesterov=True
                        )
lr_schdlr = WarmupCosineLrScheduler(optim, 
                                    n_iters_all, 
                                    warmup_iter=int(n_iters_all*0.1)
                                    )
```
We can change the patch_size (e.g., patch_size=8) for inputs with different resolutions, but smaller patch_size may not always bring performance improvements.


# Citation
```
@misc{tolstikhin2021mlpmixer,
      title={MLP-Mixer: An all-MLP Architecture for Vision}, 
      author={Ilya Tolstikhin and Neil Houlsby and Alexander Kolesnikov and Lucas Beyer and Xiaohua Zhai and Thomas Unterthiner and Jessica Yung and Daniel Keysers and Jakob Uszkoreit and Mario Lucic and Alexey Dosovitskiy},
      year={2021},
      eprint={2105.01601},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

# Acknowledgement
1. The implementation is based on the original paper <https://arxiv.org/abs/2105.01601> and the official Tensorflow repo: <https://github.com/google-research/vision_transformer>.
2. It also refers to the re-implementation repo: <https://github.com/d-li14/mlp-mixer.pytorch>.

