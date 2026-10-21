# Technical interview questions and answers

## Q1. Why is a CNN appropriate for satellite image classification?
A. Satellite scenes have spatial patterns such as roads, crop boundaries, water bodies, and residential blocks. CNNs capture local texture and spatial hierarchies better than tabular or simple image flattening approaches.

## Q2. What is the difference between a baseline CNN and transfer learning?
A. A baseline CNN learns filters from scratch. Transfer learning reuses pretrained image features from ImageNet, which can reduce training time and often improve performance when data is limited.

## Q3. Why do you keep a test set separate?
A. To avoid test leakage and produce a realistic estimate of model performance on unseen data.

## Q4. Why is Grad-CAM useful but limited?
A. It highlights influential regions, but it is an approximation and does not prove causal importance.

## Q5. What should you do when the dataset is imbalanced?
A. Use stratified splits, monitor macro metrics, and consider class-weighting or augmentation depending on the class distribution.
