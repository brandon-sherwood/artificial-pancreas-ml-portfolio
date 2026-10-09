# Artificial Pancreas: Time-Series Analysis & Machine Learning

## Overview

This repository provides an overview of a data mining and machine learning project I completed for CSE 572: Data Mining as part of my Master of Computer Science program at Arizona State University.

The project uses continuous glucose monitor (CGM) and insulin pump data to explore different data science and machine learning techniques, including time-series analysis, feature engineering, supervised classification, and unsupervised clustering.

The analysis progresses from statistical evaluation of glucose measurements to training and evaluating machine learning models.

## Dataset and Preprocessing

The analysis used two datasets: continuous glucose monitor (CGM) readings collected every five minutes and insulin pump records containing timestamps and operating events.

Using Python and pandas, I combined the separate date and time columns into timestamps and sorted the data chronologically. Since the CGM and insulin pump operated independently, their timestamps did not align exactly. I identified the first transition into Auto Mode from the insulin pump records and matched it to the next available CGM reading.

After separating the data into Manual Mode and Auto Mode, I grouped the glucose readings by day and divided them into overnight (12 AM to 6 AM), daytime (6 AM to 12 AM), and whole-day intervals.

The dataset also contained missing glucose measurements (NaN values). For the initial statistical analysis, I removed observations with missing glucose readings using pandas `dropna()` while retaining the remaining valid measurements. Rather than estimating missing values through interpolation, the statistical calculations used the available readings and the expected 288 measurements per day as the reference for calculating percentages.

## Exploratory Data Analysis

### Glucose Control Metrics

The initial analysis focused on comparing how effectively blood glucose levels were maintained during Manual Mode versus Auto Mode operation.

To compare glucose control between these operating modes, I calculated six metrics based on glucose concentration thresholds.

| Metric | Glucose Range |
|---|---|
| Hyperglycemia | > 180 mg/dL |
| Critical Hyperglycemia | > 250 mg/dL |
| Time in Range | 70–180 mg/dL |
| Secondary Time in Range | 70–150 mg/dL |
| Hypoglycemia Level 1 | < 70 mg/dL |
| Hypoglycemia Level 2 | < 54 mg/dL |

Each metric was calculated separately for overnight, daytime, and whole-day periods, resulting in 18 measurements for each operating mode.

With glucose readings collected every five minutes, a complete day contains 288 measurements. The percentage of time spent within a given glucose range was calculated by dividing the number of readings satisfying that condition by the expected 288 daily measurements and multiplying by 100.

The daily percentages were then averaged across the observation period to obtain an overall set of glucose-control metrics for Manual Mode and Auto Mode.

$$
\text{Percentage} = \frac{\text{Readings satisfying condition}}{288} \times 100
$$

### Manual vs. Auto Mode Results

The analysis produced 36 glucose-control metrics: 18 for Manual Mode and 18 for Auto Mode. These results were used to compare the percentage of time spent within different glucose ranges under each operating mode.

The following figures compare the glucose-control metrics for Manual Mode and Auto Mode across the observation period.

*Visualizations to be added.*

## Feature Engineering

### Time-Domain Features

*To be added.*

### Frequency-Domain Features (FFT)

*To be added.*

### Feature Selection

*To be added.*

## Supervised Machine Learning

### Decision Tree Classification

*To be added.*

### Model Evaluation and Results

*To be added.*

## Unsupervised Machine Learning

### KMeans and DBSCAN Clustering

*To be added.*

### Cluster Validation and Results

*To be added.*

## Key Findings and Lessons Learned

*To be added.*

## Technologies Used

*To be added.*

## Academic Integrity

*To be added.*
