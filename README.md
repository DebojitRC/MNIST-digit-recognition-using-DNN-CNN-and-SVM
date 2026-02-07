# MNIST digit recognition using DNN, CNN and SVM

Practical Application of Machine Learning Algorithm

**Author:** Debojit Roy Chowdhury<br>
**Tools Used:** Python<br>
**Dataset Source:** [Machine Learning Mastery] (https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/)

---

## Statement

Handwritten digit recognition is a widely studied problem in machine learning and serves as a benchmark for evaluating classification techniques. This project focuses on the recognition of handwritten digits using the MNIST dataset. The objective is to implement and compare three approaches—Support Vector Machine (SVM), Deep Neural Network (DNN), and Convolutional Neural Network (CNN)—under a unified experimental framework. The comparative study aims to analyse their classification performance and highlight the effectiveness of deep learning models, particularly CNNs, in image-based recognition tasks.

## Dataset Description

The MNIST dataset consists of grayscale images of handwritten digits ranging from 0 to 9. Each image has a resolution of 28×28 pixels and is associated with a corresponding digit label. The dataset contains 60,000 training samples and 10,000 testing samples. Due to its standardized format and moderate complexity, MNIST is commonly used as a benchmark dataset for evaluating classification algorithms in image recognition tasks.

## Methodology

The proposed methodology follows a systematic pipeline for handwritten digit recognition using the MNIST dataset. Initially, the dataset is loaded and pre-processed by normalizing pixel intensity values and reshaping the input data to meet model-specific requirements. The processed data are then used to train three different classifiers: Support Vector Machine (SVM), Deep Neural Network (DNN), and Convolutional Neural Network (CNN).

For the SVM model, the two-dimensional image data are flattened into one-dimensional feature vectors prior to training. In contrast, the DNN and CNN models operate directly on normalized pixel values, with CNNs additionally exploiting spatial features through convolution and pooling operations. Each model is trained and evaluated under a consistent experimental setup to ensure fair comparison.

The performance of the models is assessed using standard evaluation metrics, and the results are analysed to compare classification accuracy and overall effectiveness of the different approaches.

## Experimental Details

### Architecture Description

Three different classification models were implemented to evaluate handwritten digit recognition performance. The Support Vector Machine (SVM) model operates on flattened input images, where each 28×28 image is converted into a one-dimensional feature vector. A multiclass classification strategy is employed to distinguish between the ten-digit classes.
The Deep Neural Network (DNN) consists of fully connected layers that take normalized pixel values as input. The network includes one or more hidden layers with nonlinear activation functions, followed by an output layer with softmax activation for multiclass classification.
The Convolutional Neural Network (CNN) is designed to exploit spatial features present in image data. It comprises convolutional layers for feature extraction, pooling layers for dimensionality reduction, and fully connected layers for classification. The final layer uses softmax activation to produce class probabilities for the ten-digit classes.

### Evaluation Metrics

The performance of the implemented models is evaluated using standard classification metrics. Overall classification accuracy is used as the primary metric to measure the proportion of correctly classified test samples. In addition, confusion matrices are employed to analyse class-wise prediction performance and identify misclassification patterns among digit classes. These metrics provide a quantitative basis for comparing the effectiveness of SVM, DNN, and CNN models under a common evaluation framework.

### Dataset Split

The MNIST dataset is divided into training and testing subsets as provided in its standard configuration. A total of 60,000 samples is used for model training, while 10,000 samples are reserved for testing. This predefined split ensures consistency with existing benchmark studies and enables fair comparison of model performance.

### Implementation Details

All experiments were implemented using Python in a Jupyter Notebook environment. The models were developed using standard machine learning and deep learning libraries, including scikit-learn for SVM and TensorFlow for DNN and CNN implementations. Input images were normalized prior to training to improve convergence. Model training and evaluation were performed on a standard CPU-based system, as the dataset size and model complexity do not require specialized hardware.

### Other Methods

In addition to the primary models, standard preprocessing techniques such as data normalization were applied uniformly across all experiments. No data augmentation or advanced optimization techniques were used, in order to maintain a fair and consistent comparison between the models. This approach ensures that performance differences are primarily attributable to model architecture rather than auxiliary enhancements.

## Results and Discussion

### Results

