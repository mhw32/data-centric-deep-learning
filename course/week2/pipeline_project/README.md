# Week 2 Project: Deep Learning Workflow

This week's project will be a little bit different than last week. We won't be using Colab notebooks for the majority of this week's content (I'm sure after last week, this comes as a breath of fresh air!).

Rather we will be getting you familiar with state-of-the-art deep learning tools to reliably build, test, and deploy deep learning models.

In particular, we will be using [MetaFlow](https://metaflow.org/), [ONNX](https://onnx.ai/), and [weights and biases](https://wandb.ai/site) among many more. However, our focus is not just on the tooling! Our primary goal is to leverage these diverse tools to teach the concepts behind deploying high quality deep learning systems.

## Warmup

If you haven't yet, please do the [tutorial exercises](https://docs.metaflow.org/getting-started/tutorials) for MetaFlow. This will save you a lot of time in familiarizing yourself with the system.

## Setup and Installation

We included a `setup.sh` script that will install a virtualenv and then install necessary dependencies. This can be done on Gitpod or your personal computer. You will not need a GPU.

## Project Files

There are a few files that you will have to complete! The project page should guide you through the steps. Here is a quick summary:

- `flow.py` is the primary file. It contains a barebones scaffold we will slowly be adding to! A decent portion of the file will be commented out. Please follow the instructions on the project page to slowly complete individual sections.

- `utils.py` contains important utilities used in the flow. Significant portions of this file will require you to complete it.

- `tests/` is a directory of tests for continuous integration. When the project page informs you, you will need to complete sections in the files in this folder.
