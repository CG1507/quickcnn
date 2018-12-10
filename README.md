<h1 align="center">
  <img src="https://github.com/CG1507/quickcnn/blob/master/images/logo.png" width="900" alt="QuickCNN">
</h1>

QuickCNN is high-level library written in Python, and backed by the [Keras](https://github.com/keras-team/keras), [TensorFlow](https://github.com/tensorflow/tensorflow), and [Scikit-learn](https://github.com/scikit-learn/scikit-learn) libraries. It was developed to exercise faster experimentation with Convolutional Neural Networks(CNN). Majorly, it is intended to use the [Google-Colaboratory](https://colab.research.google.com/) to quickly play with the ConvNet architectures. It also allow to train on your local system.

#### Why Google-Colaboratory:question:

It gives you massive computing power with following specification: :free:

- GPU: 1xTesla K80, compute 3.7, having 2496 CUDA cores, 12GB GDDR5 VRAM

- CPU: 1xsingle core hyper threaded Xeon Processors @2.3Ghz i.e(1 core, 2 threads)

- RAM: ~12.6 GB Available

- Disk: ~33 GB Available

- Do not require to install any prerequisites packages

#### Go for QuickCNN, if you:

- don't have GPU, and want to train Deep ConvNet model on any size of data.

- want to apply transfer learning, finetuning, and scratch training on Deep ConvNet models.

- want to use pretrained ConvNet architectures for above learning process.

> Main idea of QuickCNN is to train deep ConvNet without diving into architectural details. QuickCNN works as an **interactive tool** for transfer learning, finetuning, and scratch training with custom datasets. It has pretrained model zoo and also works with your custom keras model architecture.

**QuickCNN works in two different environment:**

- [Google-Colaboratory](https://colab.research.google.com/) (Recommended)

- Locally

## :ocean: Prerequisites (Only for working locally)

- Keras

- Tensorflow

- sklearn

- cv2

- matplotlib

- pandas

> **Google-Colaboratory**: No such prerequisites. Already installed on Google-Colaboratory VM.

## :seedling: Installation

- Google-Colaboratory

```
!pip install -q quickcnn
```

- Locally

```
sudo pip3 install quickcnn
```

## :gem: Features

## :running: Getting started

1. **Upload Data**

  - **Colab:** Upload your dataset on Google Drive in either of following structure. 
  
  - **Locally:** Get your dataset in either of the following structure. 
  
  > NOTE: If you don not want to change original data folder structure, then make a copy of your dataset, because we will split dataset in train/validation set.

  
  If you **have splitted data** in train/validation set:
  
  ```
    ├── "dataset_name"                   
    |   ├── train
    |   |   ├── class_1_name
    |   |   |   ├── image_1
    |   |   |   ├── image_2
    |   |   |   ├── image_X
    |   |   |   ├── .....
    |   |   ├── class_2_name
    |   |   |   ├── image_1
    |   |   |   ├── image_2
    |   |   |   ├── image_X
    |   |   |   ├── .....
    |   |   ├── .....
    |   ├── val
    |   |   ├── class_1_name
    |   |   |   ├── image_1
    |   |   |   ├── image_2
    |   |   |   ├── image_X
    |   |   |   ├── .....
    |   |   ├── class_2_name
    |   |   |   ├── image_1
    |   |   |   ├── image_2
    |   |   |   ├── image_X
    |   |   |   ├── .....
    |   |   ├── .....
  ```

 If you **do not have splitted data** in train/validation set:
  
  ```
    ├── "dataset_name"                   
    |   ├── class_1_name
    |   |   ├── image_1
    |   |   ├── image_2
    |   |   ├── image_X
    |   |   ├── .....
    |   ├── class_2_name
    |   |   ├── image_1
    |   |   ├── image_2
    |   |   ├── image_X
    |   |   ├── .....
    |   ├── .....
  ```

 
## :memo: Todo

## :mag: Related Projects

## :sunglasses: Authors

## :green_heart: Acknowledgments
