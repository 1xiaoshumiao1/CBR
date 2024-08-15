## Class-Balanced Regularization for Long-Tailed Recognition 
Yuge Xu, Chuanlong Lyu
_________________

This is the implementation of CBR in the paper [Class-Balanced Regularization for Long-Tailed Recognition](https://doi.org/10.1007/s11063-024-11624-x) in PyTorch.

### Dataset

- Imbalanced [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html). The original data will be downloaded and converted by `imbalancec_cifar.py`.

### Training 

We provide several training examples:

- To train the baseline network on CIFAR-100 with an imbalance factor of 100

```bash
python cifar_train.py --arch resnet32 --dataset cifar100  --loss_type 'CE' --imb_factor 0.01 --batch_size 128 
```

- To balance the classifer on CIFAR-100 with an imbalance factor of 100

```bash
python cifar_train_classifier.py --arch resnet32 --dataset cifar100  --loss_type 'CBR' --imb_factor 0.01 --batch_size 128
```


### Citation

If you find our paper and repo useful, please cite as

```
@article{xu2024class,
  title={Class-Balanced Regularization for Long-Tailed Recognition},
  author={Xu, Yuge and Lyu, Chuanlong},
  journal={Neural Processing Letters},
  volume={56},
  number={3},
  pages={158},
  year={2024},
  publisher={Springer}
}
```

### Acknowledgements
This is a project based on [LDAM-DRM](https://github.com/kaidic/LDAM-DRW) and [GCL](https://github.com/Keke921/GCLLoss).