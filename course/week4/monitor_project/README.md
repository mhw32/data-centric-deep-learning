# Project for Week 4

This week's project simulates a model in production making predictions for streaming data. The task we are focusing on is sentiment analysis for customer reviews of Amazon products. The catch is that the data distribution is shifting. Your job is to write monitoring code to compute metrics and track this distribution shift. Once we find it, you will retrain the model on newly annotated data. 

We will also study group bias and fairness. When distribution shift happens, we might end up with two distinct groups: data from the new shifted distribution and data from the old distribution. We want our model to perform equally across both groups. To do this, we will use the "Distributionally Robust Optimization (DRO)" algorithm that we cover in the readings. This will be a chance for learners to translate an academic paper to code. There is also an optional extension for users to implement a second algorithm "Just Train Twice" and compare results.

## Instructions

Before getting started, do not forget to run `source init_env.sh`.
