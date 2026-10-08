# Architecture notes

## Baseline UNet

The baseline model is a standard UNet with:

- double convolution blocks
- four encoder down-sampling stages
- four decoder up-sampling stages
- skip connections at matching resolutions
- a final `1x1` output convolution

This model acts as the reference point for all later experiments.

## Basic model

The basic model is the first lightweight idea.

Its main characteristics are:

- the backbone keeps three main channel groups throughout the network
- each input channel is expanded with three independent `2x2` kernels
- feature processing relies heavily on depthwise convolutions
- the decoder combines skip and upsampled features as paired channel groups

This version is kept mainly as an archived design reference.

## Premier model

The premier model is the main custom architecture explored in this project.

Its core idea is to keep a compact three-group backbone while creating richer temporary features inside each block.

Main components:

1. `SamePadConv2d`
   - manual zero padding for consistent shape handling across `1x1`, `2x2`, and `3x3` kernels

2. `MultiKernelExpand`
   - expands each group into multi-scale branches
   - uses parallel `1x1`, `2x2`, and `3x3` kernels

3. `CrossGroupLinearMix`
   - allows interactions between different groups at the same scale
   - keeps the mixing constrained instead of fully dense

4. `ChannelLinearMix`
   - applies residual mixing after features are reduced back to the three main channels

5. `IndependentBlock`
   - expand into multi-scale channels
   - mix across groups
   - apply depthwise spatial processing
   - reduce back to three main channels

At the macro level, the model still follows a UNet encoder-decoder structure, but the internal feature processing is much more constrained and structured than in a standard UNet.
