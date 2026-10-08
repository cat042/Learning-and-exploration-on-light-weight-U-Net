# Experiment notes

## Project motivation

This project started as a summer interest project around medical image segmentation.

I wanted to do more than just run an existing model, so I used a standard UNet as a baseline and then tried to design lighter alternatives inspired by grouped processing and depthwise convolutions.

## What was explored

- baseline reproduction with a conventional UNet
- a first lightweight grouped design (`basic_model`)
- a refined grouped lightweight design (`premier_model`)
- several follow-up training and loss-function variations

## Result summary

The baseline UNet clearly outperformed the custom lightweight variants in the current experiments.

Even so, the project remains useful because it demonstrates:

- implementation of a full training / evaluation / prediction pipeline
- comparison against a baseline instead of presenting only a custom idea
- willingness to keep negative results and analyze them honestly
- practical experimentation with architecture and optimization choices

## Reflection

The current custom designs likely sacrifice too much representational capacity by forcing the visible backbone to stay at three channels almost everywhere.

Possible future directions include:

- relaxing the fixed three-channel backbone constraint
- introducing pointwise mixing or learned bottlenecks at more stages
- comparing parameter count and FLOPs explicitly against accuracy
- visualizing prediction failures to understand where the lightweight design breaks down

This makes the project a useful learning artifact rather than just a polished benchmark result.
