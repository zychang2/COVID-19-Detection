# COVID-19-Detection
Link to report: https://jizhou-chen.github.io/CS7641_Group18/Final 
## Introduction

The outbreak of the COVID-19 pandemic in 2019 has revealed the critical need for rapid and accurate diagnostic tools to combat highly infectious diseases. Chest X-ray images have proven valuable for diagnosing respiratory diseases, including COVID-19. 

Prior research has explored various methods including both supervised and unsupervised learning and showed their effectiveness in tackling image classification tasks in general as well as X-ray-based lung disease diagnosis specifically.

We aim to leverage the dataset collected during the pandemic to develop a robust and efficient machine-learning system for detecting COVID-19 based on chest X-ray images to enhance our preparedness for future pandemics where early and accurate diagnosis plays a crucial role. 

## Problem Definition and Motivation

The primary motivation for this project is to enhance the efficiency and accuracy of COVID-19 diagnosis to enhance our preparedness for future pandemics. Currently, the manual interpretation of chest X-ray images by radiologists is time-consuming and may lead to misdiagnoses due to human error. By developing a machine learning model that can autonomously detect COVID-19, we can:

1. Speed up the diagnostic process, enabling prompt treatment initiation.
2. Reduce the dependency on expert radiologists, especially in resource-constrained areas.
3. Enhance diagnostic accuracy, thereby improving patient outcomes.

## Data Collection

With the motivation in mind, we found an appropriate dataset for the problem we are attempting to solve. This dataset comprises over 20,000 chest X-ray images, categorized into four distinct classes: COVID-19-infected, normal, lung opacity (indicating non-COVID-related infections), and viral infected. Each class boasts an ample quantity of images, ensuring the dataset's adequacy for training purposes. Notably, the dataset includes masks delineating lung areas, a valuable addition intended to mitigate the inadvertent absorption of environmental data present in the X-ray images during the training process. Detailed descriptions of the dataset are here: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database?select=COVID-19_Radiography_Dataset.

The dataset consists of 4 classes: COVID-19 (3616 images), lung opacity (6012 images), normal (10192 images), viral pneumonia (1345 images). Every image is (300, 300, 3) that comes with a mask that selects the lung area of the image. 

Sample image in dataset:

![image](https://github.com/zychang2/COVID-19-Detection/assets/81200885/d86a4cdd-6f09-4f8e-8919-c2066402f969)

Covid image

![image](https://github.com/zychang2/COVID-19-Detection/assets/81200885/57e9fd6a-42fe-42af-b1b0-dc0dde22159d)


Normal image

![image](https://github.com/zychang2/COVID-19-Detection/assets/81200885/6e220d05-380d-41da-8583-771186537f02)


Lung Opacity

![image](https://github.com/zychang2/COVID-19-Detection/assets/81200885/08dfc39b-13f7-440d-b5e1-34ca30c7a373)


Viral Pneumonia

## Data Preprocessing

We applied the following pipeline to process our image data:

1. **For each of the four classes, we randomly selected 1200 images. This gave us a dataset of 4800 images in total with four classes of equal size.** We chose this number of data based on the maximum of our computational resources and the dataset size. One of our dataset classes only includes 1300+ images, so we want to keep the balance between classes as much as we can in order to get better results.
2. **Changed each image from RGB to Grayscale.** X-ray images are visualized data of electromagnetic radiation. Based on this property, we know that X-ray images originally contained only one-channel information instead of normal colorful images with 3 channels. However, all the images in the dataset are in PNG formats with RGB channels. So, considering both the computation efficiency and the characteristics of X-ray images, we converted all the X-ray images to Grayscale.
3. For **unsupervised learning** method and **Random Forest** method:
    1. For convolutional neural network (CNN) method: **Resized image from (300, 300) to (256, 256).** We chose 256 because CNN has strong capability in handling images.
    2. For the rest: **Resized image from (300, 300) to (64, 64).** This resizing step is for better training efficiency while not losing important information and features in the images.

**Data augmentation**

To help our model to generalize better to new data and reduce the risk of overfitting, we performed data augmentation. Specifically, with our observations 1) patients’s body positions when taking X-ray scans could affect the orientation of the X-ray images, and 2) flipped X-ray images should yield the same diagnostic results, we randomly flip and rotate the X-ray images by (-36, 36) degrees in training set with a probability of 0.2.

We chose 1200 images from each class instead of using all images because we found this is the maximum number of images we can read and store in the GPU memory since we are using Colab.

## Methods

### Unsupervised Learning

In unsupervised learning, we applied Principal Component Analysis (PCA) to reduce dimensionality as our data preprocessing, evaluating the impact of selecting 2 to 50 principal components on data representation efficiency. We flattened each image to a vector of (1, 4096) first for feature selection.

1. Flattened each image to a vector of (1, 4096).
2. Applied Principal Component Analysis (PCA) to reduce dimensionality to 50 dimensions. Then, we trained our models on the PCA-reduced image data iteratively by selecting 2~50 features to determine the best number of features yielding the best result.

**K-means, GMM**

We then clustered the dimensionality-reduced data into four categories using both K-means and Gaussian Mixture Models (GMM), examining the data's inherent grouping patterns. We chose these two methods as K-means is easy to implement, computationally efficient for large datasets and it always converges. GMM, on the other hand, provides more flexibility due to its soft-clustering (sample membership of multiple clusters) and cluster shape adaptability.

### Supervised Learning

**Random Forest**

In supervised learning, we utilized the Random Forest classifier, an ensemble learning method known for its robustness and high accuracy. We chose Random Forest as it is a robust algorithm that can handle outliers and imbalances in the data effectively, and it is less likely to overfit as it uses multiple decision trees and gives outputs based on majority voting. 

**Convolutional Neural Network (CNN)**

Addtionally, we used Convolutional Neural Network (CNN) for its advantages in automatically detecting important features without any need for manual feature extraction, as well as capturing local dependencies and spatial hierarchies in the image. Our data was partitioned into two subsets: 80% for training the model; the remaining 20% for assessing the model's predictive performance.

CNN Architecture:

![image](https://github.com/zychang2/COVID-19-Detection/assets/81200885/8f3a94e6-caf6-4b39-851c-e6710b0fd159)


Our CNN architecture started from a simple structure. Originally, we picked a relatively deeper CNN followed by 3 smaller Fully Connected Layers for faster training and inference. However, we noticed that, in this specific task, accuracy should be the first priority instead of speed. Therefore, we tuned the model structure by decreasing the CNN depth and increasing the FC layer size, in order to increase the feature size available for capturing more details from the medical images. Eventually, we reached the architecture shown above, which we have tested to give the best accuracy among all the architectures we tried.

Meanwhile, when training the CNN, we tuned the training parameters to get a better training accuracy and performance. Specifically, we tuned the number of epochs, the batch size, and the learning rate for our CNN and finally arrive at what we think achieves the best result:

> num_epochs: 30, batch_size: 16, learning_rate: 5e-3
> 

One interesting discovery is that pre-trained models, such as ResNet and VGG, are not performing better than our own CNN structure in any case. We think that it’s because the most popular pre-trained models are trained with ImageNet, which is a more general dataset instead of a medical-based one. So, transfer learning might not work because the pre-trained models didn’t learn medical features more than our CNN did.

## Results and Discussion

### K-means (Unsupervised)

![image](https://github.com/zychang2/COVID-19-Detection/assets/81200885/c65d40e3-1fbb-4d39-9554-3f9977e314b4)


![image](https://github.com/zychang2/COVID-19-Detection/assets/81200885/b22659a7-a23f-4dad-9aac-ee67cb543484)


### GMM (Unsupervised)

![image](https://github.com/zychang2/COVID-19-Detection/assets/81200885/a9e3552a-0461-4763-93dd-357d1f1ea345)


![image](https://github.com/zychang2/COVID-19-Detection/assets/81200885/7a14da36-3833-47ff-9bbc-c3205162c7f2)


### RandomForest (Supervised)

![image](https://github.com/zychang2/COVID-19-Detection/assets/81200885/999921be-e301-43e9-9f69-3b6fe032a3fd)


![image](https://github.com/zychang2/COVID-19-Detection/assets/81200885/fbe3f354-1484-4024-aa1e-344e368121df)


### CNN (Supervised)

![image](https://github.com/zychang2/COVID-19-Detection/assets/81200885/a0f153a7-878e-4838-8465-30052de99fa8)


![image](https://github.com/zychang2/COVID-19-Detection/assets/81200885/9081169f-8b5b-4b73-8b17-08440b55c33d)


To evaluate the performance of our models, we plan to use the following quantitative metrics:

1. **Cross Entropy Loss:** To measure the accuracy of probability given by the softmax layer.
2. **Prediction Accuracy:** To measure the overall usefulness of our classification.
3. **F-1 Score:** To measure the precision and recall performances.

With unsupervised learning, we achieved better performances with GMM model than K-Means.

| Model | Best Accuracy | F-1 Score |
| --- | --- | --- |
| K-Means | 40.2% | 0.400 |
| GMM | 54.3% | 0.548 |

Ideally, accuracy should increase and converge as the number of components increases (for both K-Means and GMM). However, we observed that the accuracy could drop as we add more components. We suspect that this phenomenon was due to that newly introduced features were noises to the model that undermine the performance. Considering we have 4 classes in total, which means the expected accuracy of random pick is 25%, the unsupervised models showed somewhat usefulness. But because the lung images are quite noisy and varied, it is hard for unsupervised models to accurately pick the features that can generalize through the whole dataset.

| Model | Best Accuracy | F-1 Score |
| --- | --- | --- |
| Random Forest Classifier | 77.3% | 0.771 |
| CNN | 89.4% | 0.889 |

It’s obvious that supervised models perform much better than unsupervised models, which is expected. 

Confusion matrix of CNN:

![image](https://github.com/zychang2/COVID-19-Detection/assets/81200885/5a057e1c-5498-448a-a099-8d8b024429c7)


From the confusion matrix, we can also see that our CNN model performs well on the test dataset.

## Comparison with Existing Work:

We noticed that several open-source projects on the same dataset on Kaggle claim accuracies of over 90%. Upon further investigation, we found that it turns out that they classify only two classes: normal and infected (3 types of diseases aggregated), as opposed to the fine-grained 4 classes classification that we do. During our search, we did not find a better performing Kaggle project with 4-class classification. Given the comprehensive capability (4-class classifier) and the accuracy (89.4%) of our model, we consider that our model performs considerably well.

## Conclusion

- Traditional clustering models like K-means, GMM did not work well with image data, as they were unable to capture spatial information.
- Generally, supervised learning model performed much better better than unsupervised learning model on our dataset.
- Our tuned CNN model outperformed all the rest of the models we have implemented and tested including two pretrained models.
- Validated by the results, our CNN model performed considerably well in classifying the fine-grained 4 classes of the dataset, which made our goal achieved and could hopefully make us well-equipped for potential pandemics in the future.

## References

Sharma, A., & Mishra, P. K. (2022). Covid-MANet: Multi-task attention network for explainable diagnosis and severity assessment of COVID-19 from CXR images. *Pattern Recognition*, *131*, 108826.

Kim, H. E., Cosa-Linan, A., Santhanam, N., Jannesari, M., Maros, M. E., & Ganslandt, T. (2022). Transfer learning for medical image classification: a literature review. *BMC medical imaging*, *22*(1), 69. https://doi.org/10.1186/s12880-022-00793-7 

Ji, X., Vedaldi, A., & Henriques, J. (2019). Invariant information clustering for unsupervised image classification and segmentation. *2019 IEEE/CVF International Conference on Computer Vision (ICCV)*. https://doi.org/10.1109/iccv.2019.00996; https://doi.org/10.48550/arXiv.1807.06653

Van Gansbeke, W., Vandenhende, S., Georgoulis, S., Proesmans, M., & Van Gool, L. (2020). Scan: Learning to classify images without labels. *Computer Vision – ECCV 2020*, 268–285. https://doi.org/10.1007/978-3-030-58607-2_16; https://doi.org/10.48550/arXiv.2005.12320

Rahman, T., Khandakar, A., Qiblawey, Y., Tahir, A., Kiranyaz, S., Kashem, S. B. A., ... & Chowdhury, M. E. (2021). Exploring the effect of image enhancement techniques on COVID-19 detection using chest X-ray images. *Computers in biology and medicine*, *132*, 104319. https://doi.org/10.1016/j.compbiomed.2021.104319
