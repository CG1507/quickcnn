<h1 align="center">
  <img src="https://github.com/CG1507/quickcnn/blob/master/images/logo.png" width="900" alt="QuickCNN">
</h1>

QuickCNN is high-level library written in Python, and backed by the [Keras](https://github.com/keras-team/keras), [TensorFlow](https://github.com/tensorflow/tensorflow), and [Scikit-learn](https://github.com/scikit-learn/scikit-learn) libraries. It was developed to exercise faster experimentation with Convolutional Neural Networks(CNN). Majorly, it is intended to use the [Google-Colaboratory](https://colab.research.google.com/) to quickly play with the ConvNet architectures. It also allow to train on your local system.

Main idea of QuickCNN is to train deep ConvNet without diving into architectural details. QuickCNN works as an **interactive tool** for transfer learning, finetuning, and scratch training with custom datasets. It has pretrained model zoo and also works with your custom keras model architecture.

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

- **Google-Colaboratory**

```
!pip install quickcnn
```

- **Locally**

```
sudo pip3 install quickcnn
```

## :gem: Features

- Model can predict ImageNet class and custom class altogether in single computation per image.
- Visualize training and validation graphs in Tensorboard
- Histogram, Distribution and convolutional filter images in Tensorboard
- Prediction of images and plot images with classes.
- Tensorboard support in Colab.
- 13 Pretrained architecture with benchmark results. [details](https://keras.io/applications/#documentation-for-individual-models)

## :running: Getting started

#### 1. **Upload Data**

   - **Colab:** Upload your dataset on Google Drive in either of following structure. 
   
     > - **NOTE:** Do not upload your dataset in colab-workspace(~ 33 GB), because when session expires, it will clean all the files in workspace. Session expires after 12 hours, and idle time for session is 90 minutes. It may also interrupt the VM in worst case, so you have to re-upload your dataset, and which is not good idea :grimacing:.
  
   - **Locally:** Get your dataset in either of the following structure. 
  
     > - **NOTE:** If you don not want to change original data folder structure, then make a copy of your dataset, because we will split dataset in train/validation set.

  
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
    |   ├── validation
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

#### 2. Main call

QuickCNN is working in interactive mode, and ```retrain``` is main import of library.

```python
from quickcnn import retrain
```

Create an obect of ```retrain.Retrain``` class for applying any learning process to pretrained or custom model, and also for prediction. Here is the list for the arguments of ```Retrain``` class:

| Arguments  | Description  | Default   |
| :---:       |     :---:    |    :---:   |
| **model** | • If model is None, then it will ask you to pick pretrained model in an interactive way. <br> • For custom keras model, you can pass keras **Model** object.| None |
| **target_size** | • If your model is None then target_size (input image size) is not required. <br> For custom model it is required to pass as tuple, if model doesn't have batch_input_shape with 4 dimensions (batch_size, width, height, channel). <br> • e.g. target_size=(224, 224) | None |
| **train_mode** | • It is always True for transfer-learning, training, finetuning and bottleneck features training on SVM. <br> • False when you only want to predict using model. | True |
| **train_dir_name** | • **Use if you have splitted data in train & validation sets.** <br> • If None, then full_data_dir_name must not be None. <br> • **for colab:** only directory-name (Nested directory in Google-Drive then pass as path: "dataset_name/train") <br> • **for locally:** path to directory ("/home/dell/Desktop/Food image data/train_data") | None |
| **val_dir_name** | • **Use if you have splitted data in train & validation sets.** <br> • If None, then full_data_dir_name must not be None. <br> • **for colab:** only directory-name (Nested directory in Google-Drive then pass as path: "dataset_name/train") <br> • **for locally:** path to directory ("/home/dell/Desktop/Food image data/train_data") | None |
| **full_data_dir_name** | • **Use if you do not have splitted data.** <br> • **for colab:** only directory-name (Nested directory in Google-Drive then pass as path: "dataset_name") <br> • **for locally:** path to directory ("/home/dell/Desktop/Food image data") | None |
| **fraction** | • If full_data_dir_name is not None then it will divide your data in train/validation set. <br> • default=80 means 80% images in training set and 20% in validation set. | 80 |
| **epoch** | • Training epoch in any of the learning process except SVM. | 20 |
| **batch_size** | • batch_size in training. <br> This is very important argument to GPU utilization. <br> • **Colab:** Check utlization in Manage Session <br> • If it is underutilized then increase the batch-size, and vice-versa. | 64 |
| **initial_lrate** | • Learning rate at epoch 1.  <br> • Do not apply in transfer learning. | 0.01 |
| **exp_drop** | • Every epoch, learning rate updated by: <br> **initial_lrate * e^(-exp_drop * epoch)** | 0.3 |
| **dropout** | • if 0.0 then do not use dropout layer with newly added FC layer for training data, else use with given rate. | 0. |
| **model_save_period** | • model will be saved (with weights) after given epoch period. | 1 |
| **dense_layer** | • Add given number of new FC layer for training data. | 1 |
| **cpu_workers** | • if cpu is underutilized then increase number, else vice-versa. <br> • It helps in CPU-GPU bottleneck by CPU based data augmentation. | 5 |
| **preserve_imagenet_classes** | • If you choose pretrained model, then save model with imagenet classes output. | False |
| **use_tensorboard** | • Allow tensorboard to visualize **loss**, **accuracy** and **graph** | False |
| **histogram_freq** | • Write **histograms**, **distribution**, and **images** after given epoch frequency. <br> • If histogram_freq is 0, then it won't write histograms, distribution, and images. <br> • Writing it frequently slow down your training. | 1 |
| **write_grads** | • Write gradents in tensorboard. | True |
| **write_images** | • Write images in tensorboard. | True |
| **initial_epoch** | • it helps to resume the training from the epoch, where it stopped. | 0 |
| **class_mapping** | • if train_mode is False and model has classes other than ImageNet, then pass class_mapping. <br> • class_mapping can be dictionary or file_path to class_mapping.json. | None |
| **name** | • name of model | "custom_convnet" |

<h3 align="center">“In colab, results are saved in Google Drive finetune ConvNet directory.”</h3>

**NOTE:** QuickCNN is saving **class-mapping.json** and all **model\*.hdf5** in your Google-Drive, so for re-using these files in arguments like ```model``` & ```class_mapping```, you have to append **'gdrive/My Drive/[Google-Drive path]**

**Do always:** If you want to train model one after another then after one training process **Runtime > Reset all runtimes...**, else it will give an error in tensorboard writing. If you inturrept the training and want to access tensorboard events then before resetting copy log folder from colab-workspace.

## :bullettrain_front: Training Mode:

There is four case with **size** of dataset and **similarity** with pretrained model's dataset.

**1. Small & similar dataset:**

```training_mode=1 in QuickCNN (Transfer Learning)```
Train only newly added FC layer for new classes. Finetuing may lead to overfitting, because of the small size of dataset.

**2. Small & dissimilar dataset:**

```training_mode=3 in QuickCNN (Train SVM from Bottleneck)```
We can not use Transfer Learning/Finetuning due to dissimilar dataset and that makes top layers learning irrelevant.
As we know earlier layers of Deep-ConvNet architecture learns edges and blobs, which can be helpful in most of the dataset. So we train the SVM classifier from activations of earlier layers.

**3. Big & similar dataset:**

There is two ways for this case.

- **Two-Step (Recommended)**
  1. ```training_mode=1 in QuickCNN (Transfer Learning)``` Do transfer learning with slow learning rate(SGD) to avoid rapid changes in top layers weights.
  2. ```training_mode=2 in QuickCNN (Finetuning) | model=<path-to-model> saved by first step.``` Apply finetuning with adaptive learning rate(Adam).

- **Direct**
  - ```training_mode=2 in QuickCNN (Finetuning)``` Apply direct finetuning with adaptive learning rate(Adam).

**4. Big & dissimilar dataset:**

```training_mode=4 in QuickCNN (Scratch training)```
As we have big data, so we can perform full training, and it is useful to initialize the model with pretrained weights.

## :octocat: Examples:

If you want to train any of 13 available pretrained models, then do [step-1](https://github.com/CG1507/quickcnn#1-upload-data) and follow this code-snippet.

```python
convnet = retrain.Retrain(train_dir_name = 'Food image data/train_data',
                          val_dir_name = 'Food image data/val_data', 
                          preserve_imagenet_classes=False, 
                          epoch=20, use_tensorboard=True, histogram_freq=0, batch_size=32)
```

For prediction after completing any of training process. If you are working in colab then it plots images and labels.
For ```preserve_imagenet_classes=True```, it also predict ImageNet class label.

```python
# test_data folder having mixed class images OR test.jpg
convnet.predict('test_data')
print(convnet.results)
```

For extracting features from layer:

```python
convnet = retrain.Retrain(train_dir_name = 'Food image data/train_data',
                          val_dir_name = 'Food image data/val_data')

# Bottleneck features
print(convnet.X)

# Image label
print(convnet.Y)
```

Directly predicting on model:

```python
convnet = retrain.Retrain(train_mode=False)

# test_data folder having mixed class images OR test.jpg
convnet.predict('test_data')
print(convnet.results)
```

## :memo: Todo

- [x] Tensorboard support.
- [ ] Obect detection training.
- [ ] T-SNE for layer's feature embeddings
- [ ] confusion-matrix
- [ ] Evaluate Generator
- [ ] Plot tp, fp, tn, fn images with result and confidence
- [ ] Restructure the code as per coding stanadrd. :grimacing:

## :mag: Related Projects

- [ConvNet-Zoo](https://github.com/CG1507/ConvNet-Zoo)

## :sunglasses: Authors

[<img src="https://avatars3.githubusercontent.com/u/24426731?s=460&v=4" width="70" height="70" alt="Ghanshyam_Chodavadiya">](https://github.com/CG1507)

## :green_heart: Acknowledgments

:thumbsup: [tensorboardcolab](https://github.com/taomanwai/tensorboardcolab)
