# Depression Prediction Using Actigraphy Data

## Introduction

This project focuses on predicting depression using actigraphy data collected from various devices. Actigraphy, a non-invasive method, involves monitoring activity patterns over time. The analysis involves the utilization of the pyActigraphy library and various statistical techniques to gain insights into individuals' circadian rhythms and rest-activity patterns.

## Table of Contents

- [Dataset Description](#dataset-description)
- [Data Preprocessing](#data-preprocessing)
- [Analysis](#analysis)
- [Results](#results)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Dataset Description

The dataset includes actigraphy data collected from different devices, and each row represents a timestamped activity reading. The columns include 'timestamp,' 'date,' and 'activity,' providing essential information for analysis.

![Dataset Screenshot](path/to/dataset/screenshot.png)

## Data Preprocessing

Data preprocessing involves handling missing values and normalizing features. Missing values are replaced using an auto-replacement technique, ensuring a consistent dataset. Feature normalization is performed using the window-cutting technique to omit the initial and final data, addressing zero values due to variations in device usage times.

## Analysis

### Singular Spectrum Analysis (SSA)

This analysis decomposes the actigraphy signal using Singular Spectrum Analysis (SSA). SSA is related to Principal Component Analysis (PCA) and helps quantify the variance of additive components. The resulting scree diagram visualizes partial variances, providing insights into the periodic components of the signal.

```python
# Insert code snippet for SSA analysis
