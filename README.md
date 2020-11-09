# CNN Based Car Classification
## Introduction
This project is about car classification for [stanford car dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html). The Cars dataset contains 16,185 images of 196 classes of cars and is split into 8,144 training images (avg: 41.5 images per class) and 8,041 testing images (avg: 41.0 images per class), where each class has been split roughly in a 50-50 split. Classes are typically at the level of Make, Model, Year, e.g. 2012 Tesla Model S or 2012 BMW M3 coupe.

It is difficult to directly train deep learning model on this dataset because the limited number of images. Thus we decide to use transfer learning, a common approach used in deep learning to utilize the pretrained model on [imagenet](http://www.image-net.org/) and fine-tune on our own dataset, i.e. car dataset. 

![](https://ai.stanford.edu/~jkrause/cars/class_montage.jpg)

## Image preparation 

As mentioned above, there are training images (8144) and testing images (8041). For validation purpose during the training process, I randomly split the testing data into validation and testing sets with a ratio of 1 : 4. All images were normalized before going into models.

To improve the performance and ability of the models to generalize, I applied image augmentations to **training images**, which include the followings:

- random horizontal flip
- random rotation of 15 degrees
- random horizontal and vertical shifts by 0.1
- random scaling with a range from 0.9 to 1.1

Augmented images look like below:

![](images/augmentation.png)

## Models

Three popular pre-trained models, in **PyTorch**, were used in this project, including: AlexNet, VGG19 and ResNet34. Here are the inspirations to choose these models:

- AlexNet started an era to process relatively larger color images (3 * 224 * 224), compared to previous black-white small images (1 * 32 * 32).  It was also the first time when dropout, ReLU activation function and max pooling were used in deep learning. 
- VGGNet uses smaller filters (3 * 3) and deeper networks. Compared to previous networks using large filters (11 * 11), VGGNet trains much faster.
- ResNet is a very deep network using residual connections. The residual block was brought in to deal with the vanishing gradient issue. 

## Training settings

## Results and discussions

