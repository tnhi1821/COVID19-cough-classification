# COVID19-cough-classification
## 1. Introduce
This project focuses on developing a system for classifying cough sounds related to Covid-19 into three categories: infected individuals, symptomatic individuals, and healthy individuals. Unlike previous studies that typically classify Covid-19 cough sounds against other respiratory diseases or between infected and healthy individuals, this project broadens the analysis by applying and evaluating the effectiveness of machine learning models such as Logistic Regression, Support Vector Machine, and K-Nearest Neighbors, as well as deep learning models like Neural Networks and Convolutional Neural Networks. Classifying into three categories not only enhances the accuracy in early detection of Covid-19 but also improves the practical application in disease prevention and control.

## 2. Dataset

The project uses the [COUGHVID crowdsourcing dataset](https://www.kaggle.com/datasets/nasrulhakim86/coughvid-wav) for building and evaluating the model. This is the largest publicly available cough sound dataset related to COVID-19 known as of 2021, containing nearly 35 hours of audio samples. The recordings in this dataset are in .webm or .ogg format, with a sample rate of 48 kHz, and include three health statuses: healthy, symptomatic, and Covid19.

The dataset's metadata provides three main types of information: (1) contextual information such as timestamps and the probability that a recording contains a cough sound, (2) self-reported information from users, and (3) labels provided by medical experts, including clinical assessments of the cough status in the recordings.

![](https://github.com/tnhi1821/COVID19-cough-classification/blob/main/image%20source/dataset.jpg)

## 3. Method
### 3.1. Data preprocessing
**Data Filtering**

From the over 25,000 audio recordings in the COUGHVID dataset, the team filtered out the recordings to ensure model accuracy. The samples removed include:

(1) Samples missing health status information

(2) Samples with a cough_detected value < 0.8, indicating a low likelihood of cough sound in the recording.

**Zero Padding for the Dataset**

Over 50% of the recordings are between 9.5 and 10 seconds in length. To create consistent spectrograms, the team performed zero-padding to ensure all recordings are 10 seconds long.

![](https://github.com/tnhi1821/COVID19-cough-classification/blob/main/image%20source/zero%20padding.jpg)

### 3.2. Dataset balancing

![](https://github.com/tnhi1821/COVID19-cough-classification/blob/main/image%20source/distribution%20of%20status%20value.jpg)

Noticing an imbalance among the classes in the dataset, which could adversely affect the classification model's results, the team applied several methods to balance the data as follows:

- **Under Sampling:** To preserve the actual data samples, the team balanced the dataset by keeping the number of samples for the COVID-19 label constant and downsampling the other labels to achieve balance among all classes.

- **Over Sampling with SMOTE:** The team used SMOTE to generate additional samples, creating a balanced dataset with 3,000 samples for each label. They selected 3,000 samples for the "healthy" label and augmented the other labels to match this quantity.

### 3.3. Feature extraction
To assess the similarity between two audio signals, a common approach is to use time-frequency representations. **Mel Frequency Cepstral Coefficients (MFCCs)** are audio features obtained using the frequency transform of the logarithm of the spectrum. MFCCs have a frequency resolution similar to human hearing, allowing them to capture the nonlinear auditory response to sound frequencies.

![](https://github.com/tnhi1821/COVID19-cough-classification/blob/main/image%20source/process%20to%20MFCC.jpg)

From each audio recording, the team extracted 26 MFCC coefficients. The number of samples from consecutive frames is calculated as follows: each 10-second recording has 81,920 samples due to a sample rate of 8,192 Hz, with a frame length of 512 samples and no overlapping frames. Therefore, the number of samples is 81,920 / 512 = 160.

In addition to MFCCs, the team extracted other audio features:

- **Zero-crossing Rate:** Measures the number of times the amplitude of the signal crosses zero, useful for audio classification.

- **Spectral Centroid:** Represents the average frequency of the entire audio spectrum.
Spectral Bandwidth: Measures the width of the audio spectrum, reflecting the sharpness of the sound.

- **Spectral Crest Factor:** Compares the maximum peak of the audio spectrum to its mean value, helping to identify sudden high frequencies.

- **Spectral Standard Deviation:** Measures the variability of spectral components from the mean.

- **Spectral Mean:** The average value of all spectral components.

- **Spectral Skewness:** Measures the asymmetry of the audio spectrum, reflecting whether the spectrum is skewed to the left or right.

### 3.4. Model
The team applied a range of machine learning and deep learning models to perform cough sound classification, including:

- Machine Learning Models: Logistic Regression, Support Vector Machine (SVM), and K-Nearest Neighbors (KNN).

- Deep Learning Models:

**Neural Networks**

![](https://github.com/tnhi1821/COVID19-cough-classification/blob/main/image%20source/neural%20network.jpg)

**Convolutional Neural Networks**

![](https://github.com/tnhi1821/COVID19-cough-classification/blob/main/image%20source/CNN.jpg)

For deep learning models like Neural Networks and CNNs, the team trained the models for 100 epochs with a batch size of 128 to ensure effective learning from the data. After each epoch, the model's performance was evaluated by measuring accuracy on the validation dataset, with a validation_split of 0.15. This process helps the team monitor and adjust the model to achieve the best performance during training.

## 4. Results

![](https://github.com/tnhi1821/COVID19-cough-classification/blob/main/image%20source/result.jpg)

For dataset balancing using under-sampling, the SVM model achieved the highest accuracy among the models, with recall values showing minimal variation across labels. However, overall, the experimental results with this method were not exceptionally high for any of the trained models.

In contrast, when using the SMOTE method for over-sampling, the accuracy results for the models were relatively stable, with Logistic Regression yielding the lowest accuracy at 0.55. The KNN model demonstrated the highest recall for the COVID-19 class (0.96), but it had significant prediction errors for the Healthy class. The CNN model achieved an accuracy of 0.7 and had high recall scores across classes, particularly for the COVID-19 class (0.93).

In summary, the CNN model with the SMOTE data balancing method performed the best for the project's objective of classifying COVID-19 cough sounds.
