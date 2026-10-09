# Artificial Pancreas Data Mining Portfolio
This repository provides an overview of three projects I completed for CSE 572: Data Mining as part of my Master of Computer Science program at Arizona State University. 
The projects use continuous glucose monitor (GCM) and insulin pump data to explore different data science and machine learning techniques, including time series analysis, feature engineering, supervised classification, and unsupervised clustering. 
Each project builds concepts from the previous one, progressing from a statistical analysis of glucose measurements to training and evaluating machine learning models.

## CGM Data Analysis

### Overview
This analysis focused on continuous glucose monitor (CGM) and insulin pump data collected from an artificial pancreas system. The objective was to compare how effectively blood glucose levels were maintained during Manual Mode compared to Auto Mode operation. 

The data consisted of glucose measurements recorded every 5 minutes and separate insulin pump records containing timestamps and operating events. Since these came from two different devices the datasets needed to be synchronized before they could be analyzed. 

The analysis examined glucose levels across different times of day and calculated statistical metrics to compare the two operating modes. 

### Data and Preprocessing
The analysis used two datasets: CGM readings collected every five minutes and insulin pump records containing timestamps and operating events. 

Using Python and pandas, I combined the separate date and time columns into timestamps and sorted the data chronologically. Since the CGM and insulin pump operated independently, their timestamps did not align exactly. I identified the first transition into Auto Mode from the insulin pump records and matched it to the next available CGM reading.

After separating the data into Manual Mode and Auto Mode, I grouped the glucose readings by day and divided them into overnight (12AM to 6AM), daytime (6AM to 12AM), and whole day intervals. The CGM dataset also contained missing glucose readings (NaN values), making incomplete sensor coverage an important consideration when calculating daily statistics.

The dataset also contained missing glucose measurements (NaN values). I removed observations with missing glucose readings using pandas dropna() while retaining the remaining valid measurements. Rather than estimating missing values through interpolation, the statistical calculations used the available readings and the expected 288 measurements per day as the reference for calculating percentages.

### Methods and Metrics
To compare glucose control between Manual Mode and Auto Mode, I calculated six metrics based on glucose concentration thresholds.

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

The percentage was calculated as $P = \frac{n}{288} \times 100$.

### Results

## Machine Model Training

### Overview

### Methods

### Results

## Cluster Validation

### Overview

### Methods

### Results
