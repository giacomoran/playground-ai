# Implementing SimCLR

[paper](https://arxiv.org/abs/2002.05709)

## Notes

- Data augmentation: random cropping and flip (followed by resize back to the original size), random color distorsions, random Gaussian blur.
- Base encoder is the output after the average pooling layer of ResNet. By default they use ResNet-50.
- 2-layer MLP projection head to project the representation to a 128-dimensional latent space.
- NT-Xent loss.
- Batch size N from 256 to 8192. By default batch size 4096 and train for 100 epochs.
- LARS optimizer for all batch sizes (since learning might be unstable with SGD). Learning rate of 4.8 (= 0.3 × BatchSize/256) and weight decay of 10−6. Linear warmup for the first 10 epochs, and decay the learning rate with the cosine decay schedule without restarts.
- They use global batch norm in DDP. In contrastive learning, the affine transformation in the batch norm layer can learn to scale the dimensions along which the two positive samples are correlated, but at test time (or in other batches) you get different batch statistics. This can be ameliorated by using larger effective batches.
- They train on the original ImageNet (1.2M training images).

## Variant implemented here

The paper reports results on CIFAR-10 across batch size, and temperature (averaged over {0.5, 1.0, 1.5} learning rates). The best top-1 accuracy is achieved with configuration:

- Batch size 2048
- Temperature 0.5 when training for <300 epochs, 0.1 when training for >300 epochs

On CIFAR-10 they make the following modifications:

- In ResNet-50, they replace the first 7x7 Conv of stride 2 with 3x3 Conv of stride 1, and also remove the first max pooling operation
- Remove Gaussian Blur from the data augmentations, use 0.5 as color distortion strenght

I've also replaced batch norm in ResNet with group normalization (with 32 group). The reason is the same as for why the paper uses global batch norm: in small batches, positive pairs influence the batch statistics (by e.g. pulling the batch mean towards their regions), this might lower the loss without the network learning invariance. We train on a single device, but use gradient accumulation with microbatches, therefore we are still dealing with small batch sizes.
