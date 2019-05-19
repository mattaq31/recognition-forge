# Skip-Thought Vectors

We use the original implementation [Skip-Thought Vectors](http://arxiv.org/abs/1506.06726) as per author Ryan Kiros. The original codebase has been adjusted to fit our problem. We primarily use the decoder architecture. The original code was written in Python 2.7, and has been updated for Python 3+.

## train_skipthoughts.py

This file trains a decoder on our dataset. It has as dependencies all the files as per the original paper (we use the original pretrained encoder to vectorize our data). These dependencies, and the subsequent trained model, should be saved in `skipthought_dir/aux_data/`.

## skipthoughts_generator.py

This file is uses our dataset-specific decoder to generate and infer new recommendations, using the pretrained encoder as well. It returns the results as a dictionary depending on the requested beam-size and stochasticity variables.


