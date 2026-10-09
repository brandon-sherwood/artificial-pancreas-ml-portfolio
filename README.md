# Artificial Pancreas: Time-Series Analysis & Machine Learning

## Overview

This repository provides an overview of a data mining and machine learning project I completed for CSE 572: Data Mining as part of my Master of Computer Science program at Arizona State University.

The project uses continuous glucose monitor (CGM) and insulin pump data to explore different data science and machine learning techniques, including time-series analysis, feature engineering, supervised classification, and unsupervised clustering.

The analysis progresses from statistical evaluation of glucose measurements to training and evaluating machine learning models.

## Dataset and Preprocessing

### CGM and Insulin Pump Data

The analysis used two datasets: CGM readings collected every five minutes and insulin pump records containing timestamps and operating events.

Using Python and pandas, I combined the separate date and time columns into timestamps and sorted the data chronologically. Since the CGM and insulin pump operated independently, their timestamps did not align exactly. I identified the first transition into Auto Mode from the insulin pump records and matched it to the next available CGM reading.

After separating the data into Manual Mode and Auto Mode, I grouped the glucose readings by day and divided them into overnight (12 AM to 6 AM), daytime (6 AM to 12 AM), and whole-day intervals.

The dataset also contained missing glucose measurements (NaN values). For the initial statistical analysis, I removed observations with missing glucose readings using pandas `dropna()` while retaining the remaining valid measurements. Rather than estimating missing values through interpolation, the statistical calculations used the available readings and the expected 288 measurements per day as the reference for calculating percentages.

### Meal and No-Meal Window Extraction

for the supervised machine learning portion of the project I used CGM and insulin pump data from two patients to identify periods associated with meal consumption and periods without meals. 

I identified meal events using the carbohydrate entries recorded by the insulin pump, excluding entries with missing or zero carbohydrate values. To avoid overlapping meal periods I excluded meal events that were followed by another recorded meal within two hours. 

For each remaining meal event, I extracted a CGM observation window starting 30 minutes before the meal and ending two hours afterward. Each meal window contained 30 glucose measurements sampled at five minute intervals.

For No-Meal periods I extracted two hour windows beginning at least two hours after a recorded meal, provided there was sufficient time before the next meal. Each No-Meal window contained 24 glucose measurements. 

I then removed any windows that did not contain the expected number of readings or had missing glucose values. Unlike the initial statistical analysis, which retained individual valid readings, this step required complete observation windows to ensure consistent input dimensions for feature extraction. 

The resulting Meal and No-Meal windows were used to calculate the time domain and frequency domain features for the classification model. 

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

After completing the initial statistical analysis, I used the extracted Meal and No-Meal observation windows to build a supervised classification model. Since the raw CGM measurements were time series, I used feature engineering to represent each window using a set of numerical values describing the glucose response.

### Time-Domain Features
To prepare the CGM data for machine learning I extracted numerical features that described how glucose levels change over time. These features allow a classification model to identify patterns associated with meal consumption rather than relying on the raw glucose measurements alone. 

Four time domain features were extracted:

#### Time to Peak ($\tau$)

This measures the time required for the glucose concentration to reach its max value within the observation window. Since the CGM records measurements every 5 minutes, the time to peak was calculated as:

$$
\tau = i_{\max}\times5
$$

where $i_{\max}$ is the index of the highest glucose reading and $\tau$ is measured in minutes.

#### Normalized Peak Difference

This measures the relative increase in glucose concentration from the beginning of the observation window to its max value.

$$
\Delta G_{\text{norm}}=\frac{G_{\max}-G_0}{G_0}
$$

Normalizing by the initial glucose concentration allows glucose responses with different starting values to be compared.

#### Maximum First Difference

This feature captures the largest increase between consecutive glucose measurements, which were recorded at five-minute intervals.

$$
D_1=\max(\Delta G_t),\qquad \Delta G_t=G_{t+1}-G_t
$$

It provides a measure of how rapidly glucose increases during the observation window.

#### Maximum Second Difference

The second difference measures how the change in glucose concentration varies between consecutive intervals.

$$
D_2=\max(\Delta^2G_t),\qquad \Delta^2G_t=G_{t+2}-2G_{t+1}+G_t
$$

This captures changes in the slope of the glucose response, providing additional information about the shape of the time series. For both features, the maximum is taken over all calculated differences within the observation window.

### Frequency-Domain Features (FFT)

In addition to time-domain features, I used NumPy's Fast Fourier Transform (`np.fft.fft`) to transform the CGM time series into the frequency domain. This allowed me to identify dominant frequency components that describe how glucose levels vary over time.

I calculated the magnitude of each Fourier coefficient and normalized it by the number of samples:

$$
M_k=\frac{|X_k|}{N}
$$

where \(X_k\) represents the complex Fourier coefficient at frequency index \(k\), and \(N\) is the number of glucose measurements in the observation window.

The zero-frequency component (DC), which represents the signal's average glucose level, was excluded. I also retained only positive frequencies to avoid the redundant negative-frequency components present in the Fourier transform of a real-valued signal.

From the remaining frequency components, I selected the two with the largest magnitudes and extracted four features:

1. Magnitude of the strongest frequency component
2. Frequency of the strongest component
3. Magnitude of the second-strongest component
4. Frequency of the second-strongest component

These features provide information about the relative strength and frequency of glucose fluctuations, complementing the time-domain features that describe glucose peaks and rates of change.

### Feature Selection

After extracting the four time domain features and four frequency domain features, each Meal and No-Meal observation window was represented by eight numerical features.

For the final supervised classification model I removed the two FFT frequency features and retained the six remaining features: time to peak, normalized peak difference, maximum first difference, maximum second difference, and the magnitudes of the two strongest positive frequency components. 

This reduced the feature matrix from eight columns to six, which were then used to train and evaluate the Decision Tree classifier.

## Supervised Machine Learning

### Decision Tree Classification

After extracting and selecting the six features I used a Decision Tree classifier from scikit-learn to classify each CGM observation window as either Meal or No-Meal.

The dataset was divided into training and testing sets using an 80/20 split. I used stratified sampling to preserve the proportion of Meal and No-Meal observations in both sets and set raondom_state=42 to make the split reproducible. 

The Decision Tree model learns a series of decision rules based on the input features to distinguish between the two classes. Each internal node evaluates a feature against a threshold and the resulting branches lead to a predicted classification. 

After evaluating the model on the held out test set I retrained the classifier using the complete feature dataset and saved the trained model using Python's pickle module. 

I also developed a separate testing script that loads new CGM observation windows, applies the same feature extraction and selection process used during training, and generates predictions using the saved Decision Tree model. The predictions were then exported to a CSV file for evaluation.

### Model Evaluation and Results

To evaluate the Decision Tree classifier I used the 20% test set that was excluded from training. The model's predictions were compared against the actual Meal and No-Meal labels using two performance metrics: accuracy and F1 score.

Accuracy measures the proportion of observations that were classified correctly. 

$$
\text{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}
$$

F1 score combines precision and recall into a single metric using their harmonic mean. 

$$
F_1=2\times\frac{\text{Precision}\times\text{Recall}}{\text{Precision}+\text{Recall}} = \frac{2TP}{2TP+FP+FN}
$$

Accuracy provides an overall measure of classification performance, while F1 score helps evaluate how effectively the model identifies Meal periods while accounting for both false positives and false negatives.

The trained model was also used to generate predictions for a separate set of CGM observation windows using the testing script.

*Add performance results and visualizations later*

## Unsupervised Machine Learning

After completing the supervised classification analysis I explored unsupervised machine learning techniques to identify patterns in glucose responses associated with meal consumption. 

Unlike the Decision Tree classifier, which was trained using known Meal and No-Meal labels, this analysis used KMeans and DBSCAN clustering to group meal associated CGM observations based on similarities in their extracted features.

For this analysis I used CGM and insulin pump data from one patient. I extracted 30-reading meal windows and reused all eight features developed previously, including the two FFT frequency features that were excluded from the supervised classification model.

Unlike the supervised learning portion I used interpolation to fill missing glucose readings within otherwise complete observation windows, while discarding windows that contained no valid glucose measurements. 

Before applying the clustering algorithms, I standardized the eight features using scikiti-learn's StandardScalar to prevent features with larger numerical scales from dominating the distance calculations. 

The resulting clusters were evaluated against carbohydrate quantities recorded by the insulin pump to investigate whether similar glucose responses corresponded to similar meal carbohydrate amounts. 

### KMeans and DBSCAN Clustering

I applied two unsupervised learning algorithms, KMeans and DBSCAN, to the standardized feature dataset to identify patterns in glucose responses associated with meal consumption.

#### KMeans Clustering 

KMeans partitions the dataset into a specified number of clusters by assigning observations to the nearest cluster centroid. The algorithm iteratively updates the centroids to minimize the sum of squared distances between observations and their assigned cluster centers.

I used scikit-learn's `KMeans` implementation with six clusters, corresponding to the number of carbohydrate categories used for subsequent cluster validation. I also set `random_state=42` for reproducibility and `n_init=10` to initialize the algorithm multiple times.

#### DBSCAN Clustering

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) identifies clusters based on regions of high data density rather than distances to predefined centroids. Unlike KMeans, DBSCAN does not require specifying the number of clusters beforehand and can identify observations that do not belong to a sufficiently dense region as noise.

I used scikit-learn's `DBSCAN` implementation with `eps=1.2` and `min_samples=5`. These parameters control the neighborhood radius and minimum number of neighboring observations required to identify dense regions.

My implementation also included an additional step to handle situations where DBSCAN produced fewer clusters than the number of carbohydrate categories. In those cases, I calculated the sum of squared errors (SSE) for each cluster and used KMeans to split the cluster with the largest SSE. This process was repeated until the desired number of clusters was reached, while leaving DBSCAN's noise observations unassigned.

Both clustering approaches were then evaluated using SSE, entropy, and purity to compare cluster compactness and their relationship to recorded carbohydrate quantities.

### Cluster Validation and Results

*To be added.*

## Key Findings and Lessons Learned

*To be added.*

## Technologies Used

*To be added.*

## Academic Integrity

*To be added.*
