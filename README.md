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

| Arguments  | Description  | Default   |
| :---:       |     :---:    |    :---:   |
| model | If model is None, then it will ask you to pick pretrained model in an interactive way. <br> For custom keras model, you can pass keras [Model] object.| None |
| target_size | If you model is None then it is not required(None). <br> For custom model it is required to pass as tuple, if model doesn't have batch_input_shape with 4 dimensions (batch_size, width, height, channel). <br> e.g. target_size=(224, 224) | None |
| train_mode | It is always True for transfer-learning, training, finetuning and bottleneck features training on SVM. <br> False when you only want to predict using model. | True |
| train_dir_name | Only allow if you have splitted data in train/validation set. <br> If None, then full_data_dir_name must not be None. <br> for colab: only directory-name (Nested directory in Google-Drive then pass as path: "datset_name/train") <br> for locally: path to directory ("/home/dell/Desktop/Food image data/train_data") | None |
| val_dir_name | Only allow if you have splitted data in train/validation set. <br> If None, then full_data_dir_name must not be None. <br> for colab: only directory-name (Nested directory in Google-Drive then pass as path: "datset_name/train") <br> for locally: path to directory ("/home/dell/Desktop/Food image data/train_data") | None |
| full_data_dir_name | Only allow if you do not have splitted data. <br> for colab: only directory-name (Nested directory in Google-Drive then pass as path: "datset_name") <br> for locally: path to directory ("/home/dell/Desktop/Food image data") | None |
| fraction | If full_data_dir_name is not None then it will divide your data in train/validation set. <br> default=80 means 80% images in training set and 20% in validation set. | 80 |
| epoch | Training epoch in any of the learning process except SVM. | 20 |
| batch_size | batch_size in training. <br> This is very important argument to GPU utilization. <br> (Colab: Check utlization in Manage Session) <br>If it is underutilized then increase the batch-size, and vice-versa. | 64 |
| initial_lrate | Do not apply in transfer learning | 0.01 |
| exp_drop | Every epoch: LR updated by: initial_lrate * e^(-exp_drop * epoch) | 0.3 |
| dropout | if 0.0 then do not use dropout layer with newly added FC layer for training data, else use with given rate. | 0. |
| model_save_period | model will be saved (with weights) after given epoch period | 1 |
| dense_layer | Add given number of new FC layer for training data | 1 |
| cpu_workers | if cpu is underutilized then increase number, else vice-versa. <br> It helps in data augmentation. | 5 |
| preserve_imagenet_classes | If you choose pretrained model, then save model with imagenet classes output. | False |
| use_tensorboard | Allow tensorboard to visualize loss, accuracy and graph | False |
| histogram_freq | Write histograms, distribution, and images after given epoch frequency. <br> If histogram_freq is 0, then it won't write histograms, distribution, and images. <br> Writing it frequently slow down your training. | 1 |
| write_grads | Write gradents in tensorboard | True |
| write_images | Write images in tensorboard | True |
| initial_epoch | it helps to resume the training from the epoch, where it stopped | 0 |
| class_mapping | if train_mode is False and model has classes other than ImageNet, then pass class_mapping. <br> class_mapping can be dictionary or file_path to class_mapping.json. | None |
| name | name of model | "custom_convnet" |

```
convnet = retrain.Retrain(model=None, train_dir_name ='Food image data/train_data',val_dir_name = 'Food image data/val_data', preserve_imagenet_classes=False, epoch=1, dropout=0.0, dense_layer=1, use_tensorboard=True, histogram_freq=0, batch_size=32)
```
 
## :memo: Todo

- [ ] Tensorboard support 
- [ ] 
- [ ] 
- [ ] Restructure the code as per stanadrd

## :mag: Related Projects

## :sunglasses: Authors

[<img src="https://avatars3.githubusercontent.com/u/24426731?s=460&v=4" width="200" height="200" alt="Ghanshyam_Chodavadiya">](https://github.com/CG1507)

## :green_heart: Acknowledgments

[tensorboardcolab](https://github.com/taomanwai/tensorboardcolab)
