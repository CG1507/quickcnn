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

```python
from quickcnn import retrain
```

```
model [default=None]: If model is None, then it will ask you to pick pretrained model in an 
                      interactive way. For custom keras model, you can pass keras [Model] object.

target_size [default=None]: If you model is None then it is not required(None). For custom model it is 
                            required to pass as tuple, if model doesn't have batch_input_shape with 4 
                            dimensions (batch_size, width, height, channel). e.g. target_size=(224, 224) 

train_mode [default=True]: It is always True for transfer-learning, training, finetuning and bottleneck 
                           features training on SVM. False when you only want to predict using model.

Check your data format above to use follwing arguments:
   for colab: only directory-name (Nested directory in Google-Drive then pass as path: "datset_name/train")
   for locally: path to directory ("/home/dell/Desktop/Food image data/train_data")
   - train_dir_name [default=None]: Only allow if you have splitted data in train/validation set.
                                    If None, then full_data_dir_name must not be None.

   - val_dir_name [default=None]: Only allow if you have splitted data in train/validation set.
                                  If None, then full_data_dir_name must not be None.

   - full_data_dir_name [default=None]: Only allow if you do not have splitted data.

fraction [deafult=80]: If full_data_dir_name is not None then it will divide your data in train/validation set.
                       default=80 means 80% images in training set and 20% in validation set.

epoch [default=20]: Training epoch in any of the learning process except SVM.

batch_size [defalut=64]: batch_size in training.
                         This is very important argument to GPU utilization. 
                         (Colab: Check utlization in Manage Session)
                         If it is underutilized then increase the batch-size, and vice-versa.

Learning Rate Decay handling
    - initial_lrate [default=0.01]: 
    
    - exp_drop [defalut=0.3]:

dropout
model_save_period
dense_layer
cpu_workers
preserve_imagenet_classes
use_tensorboard
histogram_freq
write_grads
write_images
initial_epoch
class_mapping
name
```

```python
convnet = retrain.Retrain(model=None, train_dir_name ='Food image data/train_data',val_dir_name = 'Food image data/val_data', preserve_imagenet_classes=False, epoch=1, dropout=0.0, dense_layer=1, use_tensorboard=True, histogram_freq=0, batch_size=32)
```
 
## :memo: Todo

## :mag: Related Projects

## :sunglasses: Authors

## :green_heart: Acknowledgments
