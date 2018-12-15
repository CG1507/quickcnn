import os
import sys
import glob
import time
import json
import shutil
import datetime
import math
import numpy as np
import tensorflow as tf
from . import data_split
from . import setup_tbc
from . import tbc_callback
from . import ckpt_callback
from . import hist_callback
from . import write_result

import keras.backend as K
from sklearn.svm import SVC
from keras.applications import *
from keras.preprocessing import image
from keras.optimizers import SGD
from keras.layers import Dense, Activation, Reshape, Conv2D, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
from keras.models import Model, load_model

class Retrain():

	def __init__(self, model=None, target_size=None, train_dir_name=None, val_dir_name=None, train_mode=True, class_mapping=None,  
				full_data_dir_name=None, fraction=80, epoch=20, batch_size=64, initial_lrate=0.01, exp_drop=0.3, dropout=0., 
				model_save_period=1, dense_layer=1, cpu_workers=5, preserve_imagenet_classes=False, use_tensorboard=False, 
				histogram_freq=1, write_grads=True, write_images=True, initial_epoch=0, name='custom_convnet'):

		# Set arguments
		self.nb_epoch = epoch
		self.batch_size = batch_size
		self.fraction = fraction
		self.initial_lrate = initial_lrate
		self.exp_drop = exp_drop
		self.model_save_period = model_save_period
		self.cpu_workers = cpu_workers
		self.dropout = dropout
		self.dense_layer = dense_layer
		self.use_tensorboard = use_tensorboard
		self.histogram_freq = histogram_freq
		self.write_grads = write_grads
		self.write_images = write_images
		self.initial_epoch = initial_epoch
		self.train_mode = train_mode
		self.preserve_imagenet_classes = preserve_imagenet_classes
		self.class_mapping = class_mapping

		self.model_dict = {
							1: 'Xception',
							2: 'VGG16',
							3: 'VGG19',
							4: 'ResNet50',
							5: 'InceptionV3',
							6: 'InceptionResNetV2',
							7: 'MobileNet',
							8: 'MobileNetV2',
							9: 'DenseNet121',
							10: 'DenseNet169',
							11: 'DenseNet201',
							12: 'NASNetMobile',
							13: 'NASNetLarge'
							}

		# Prepare Google Drive for data
		self.is_colab = 'google.colab' in sys.modules
		if self.is_colab:
			print('-'*80)
			print('You have to mount your Google drive where you have uploaded your data.')
			print('-'*80, '\n')

			from google.colab import drive
			drive.mount('/content/gdrive')
		else:
			print('-'*80)
			print("\nLocally working...")
			print('-'*80, '\n')

		if model == None:
			self.select_model('Choose the number for ConvNet architecture from above list: ')
			self.set_model()
			self.name = self.model_dict[self.model_no]
			self.is_custom_model = False
		else:
			if type(model) == str:
				if os.path.exists(model):
					self.model = load_model(model)
				else:
					raise ValueError("Model path doesn't exist.")
			else:
				self.model = model

			if name not in self.model_dict.values():
				self.name = name
			else:
				self.name = 'custom_convnet'

			self.is_custom_model = True

			from keras.applications.imagenet_utils import preprocess_input
			print("NOTE: This model will use standard ImageNet preprocess function")
			self.preprocess_fun = preprocess_input

			if self.preserve_imagenet_classes or (not self.train_mode):
				from keras.applications.imagenet_utils import decode_predictions
				print("NOTE: This model will use standard ImageNet decoding function")
				self.decode_fun = decode_predictions

			if target_size == None:
				if K.image_dim_ordering() == 'tf':
					if type(self.model.layers[0]).__name__ == "InputLayer":
						input_layer_shape = self.model.layers[0].batch_input_shape
						if len(input_layer_shape) == 4:
							self.target_size = (input_layer_shape[1], input_layer_shape[2])
						else:
							raise ValueError("target_size is not given")
					else:
						raise ValueError("target_size is not given")
				else:
					raise ValueError("target_size is not given")
			else:
				self.target_size = target_size

		# Get training and validation data directory path
		if self.train_mode:
			if self.is_colab:
				if full_data_dir_name == None:
					if train_dir_name == None:
						raise ValueError("If full_data_dir_name is None, then train_dir_name and val_dir_name must not be None")

					if val_dir_name == None:
						raise ValueError("If full_data_dir_name is None, then train_dir_name and val_dir_name must not be None")

					self.train_dir_name = 'gdrive/My Drive/' + train_dir_name + '/'
					self.val_dir_name = 'gdrive/My Drive/' + val_dir_name + '/'
					self.full_data_dir_name = None
					self.given_data = 'splitted'

					if not os.path.exists(self.train_dir_name):
						raise ValueError("train_dir_name doesn't exist in Google Drive: ", self.train_dir_name)

					if not os.path.exists(self.val_dir_name):
						raise ValueError("val_dir_name doesn't exist in Google Drive: ", self.val_dir_name)
				else:
					self.full_data_dir_name = 'gdrive/My Drive/' + full_data_dir_name + '/'
					self.train_dir_name = None
					self.val_dir_name = None
					self.given_data = 'full'

					if not os.path.exists(self.full_data_dir_name):
						raise ValueError("full_data_dir_name doesn't exist in Google Drive: ", self.full_data_dir_name)
			else:
				if full_data_dir_name == None:
					if train_dir_name == None:
						raise ValueError("If full_data_dir_name is None, then train_dir_name and val_dir_name must not be None")
					
					if val_dir_name == None:
						raise ValueError("If full_data_dir_name is None, then train_dir_name and val_dir_name must not be None")

					self.train_dir_name = train_dir_name
					self.val_dir_name = val_dir_name
					self.full_data_dir_name = None
					self.given_data = 'splitted'

					if not os.path.exists(self.train_dir_name):
						raise ValueError("train_dir_name doesn't exist: ", self.train_dir_name)

					if not os.path.exists(self.val_dir_name):
						raise ValueError("val_dir_name doesn't exit: ", self.val_dir_name)
				else:
					self.full_data_dir_name = full_data_dir_name
					self.train_dir_name = None
					self.val_dir_name = None
					self.given_data = 'full'

					if not os.path.exists(self.full_data_dir_name):
						raise ValueError("full_data_dir_name doesn't exist: ", self.full_data_dir_name)

			self.get_splitted_data()
			self.read_splitted_data()
			self.prepare_generator()
			self.create_result_dir()
			self.save_class_mapping()
			self.prepare_new_model()
			self.train()

	def get_splitted_data(self):
		if self.given_data == 'full':
			self.train_dir_name, self.val_dir_name = data_split.split(self.full_data_dir_name, self.fraction)
			self.given_data = 'splitted'

	def get_nb_files(self, directory):
		if not os.path.exists(directory):
			return 0
		cnt = 0
		for r, dirs, files in os.walk(directory):
			for dr in dirs:
				cnt += len(glob.glob(os.path.join(r, dr + "/*")))
		return cnt

	def read_splitted_data(self):
		self.nb_train_samples = self.get_nb_files(self.train_dir_name)
		self.nb_classes = len(glob.glob(os.path.join(self.train_dir_name, '*')))
		self.nb_val_samples = self.get_nb_files(self.val_dir_name)

	def prepare_generator(self):
		train_datagen =  ImageDataGenerator(
				preprocessing_function=self.preprocess_fun,
				rotation_range=30,
				width_shift_range=0.2,
				height_shift_range=0.2,
				shear_range=0.2,
				zoom_range=0.2,
				horizontal_flip=True
		)

		test_datagen = ImageDataGenerator(
				preprocessing_function=self.preprocess_fun,
				rotation_range=30,
				width_shift_range=0.2,
				height_shift_range=0.2,
				shear_range=0.2,
				zoom_range=0.2,
				horizontal_flip=True
		)

		self.train_generator = train_datagen.flow_from_directory(
			self.train_dir_name,
			target_size=self.target_size,
			batch_size=self.batch_size
		)

		self.validation_generator = test_datagen.flow_from_directory(
			self.val_dir_name,
			target_size=self.target_size,
			batch_size=self.batch_size
		)

	def create_result_dir(self):
		if self.is_colab:
			date_time = datetime.datetime.fromtimestamp(int(time.time())).strftime('%Y-%m-%d %H:%M:%S')
			self.result_dir = 'gdrive/My Drive/finetune ConvNet/' + self.name + '/' + str(date_time)
			if not os.path.exists(self.result_dir):
				os.makedirs(self.result_dir)

	def save_class_mapping(self):
		self.class_mapping = self.train_generator.class_indices
		self.inv_class_mapping = {v: k for k, v in self.class_mapping.items()}

		if self.is_colab:
			with open(os.path.join(self.result_dir, 'class-mapping.json'), 'w') as fp:
				json.dump(self.class_mapping, fp)
		else:
			with open('class-mapping.json', 'w') as fp:
				json.dump(self.class_mapping, fp)

	def prepare_new_model(self):
		note = "Choose the training mode number: "
		self.select_training_mode(note)

		if self.is_custom_model:
			if self.training_mode == 1:
				if self.preserve_imagenet_classes:
					output_tensors = self.model.outputs
					if len(output_tensors) == 1:
						if output_tensors[0].shape[1] != 1000:
							raise ValueError("Pretrained Model doesn't have ImageNet output tensor."
										"Set: preserve_imagenet_classes = False")
						else:
							self.setup_to_transfer_learn()
							self.add_new_last_layer()
					else:
						if len(output_tensors) != 2:
							raise ValueError("Pretrained Model should have only two output tensors. "
												"One for ImageNet output, Second for new classes.")

						if output_tensors[0].shape[1] == 1000:
							if output_tensors[1].shape[1] == self.nb_classes:
								self.setup_to_transfer_learn()
								tensor_name = output_tensors[1].name
								if type(self.model.get_layer(tensor_name.split('/')[0])).__name__ != 'Reshape':
									self.model.get_layer(tensor_name.split('/')[0]).trainable = True
								else:
									self.model.get_layer(tensor_name.split('/')[0]).trainable = True
									last_second_layer = self.model.get_layer(tensor_name.split('/')[0]).input.name.split('/')[0]
									self.model.get_layer(last_second_layer).trainable = True
									last_third_layer = self.model.get_layer(last_second_layer).input.name.split('/')[0]
									self.model.get_layer(last_third_layer).trainable = True
								self.new_model = Model(inputs=[self.model.layers[0].input], outputs=[output_tensors[1]])
								self.model = Model(inputs=[self.model.layers[0].input], outputs=[output_tensors[0]])
							else:
								raise ValueError("Pretrained Model's output tensors doesn't match number of classes in last layer.")
						else:
							if output_tensors[0].shape[1] == self.nb_classes:
								self.setup_to_transfer_learn()
								tensor_name = output_tensors[0].name
								if type(self.model.get_layer(tensor_name.split('/')[0])).__name__ != 'Reshape':
									self.model.get_layer(tensor_name.split('/')[0]).trainable = True
								else:
									self.model.get_layer(tensor_name.split('/')[0]).trainable = True
									last_second_layer = self.model.get_layer(tensor_name.split('/')[0]).input.name.split('/')[0]
									self.model.get_layer(last_second_layer).trainable = True
									last_third_layer = self.model.get_layer(last_second_layer).input.name.split('/')[0]
									self.model.get_layer(last_third_layer).trainable = True
								self.new_model = Model(inputs=[self.model.layers[0].input], outputs=[output_tensors[0]])
								self.model = Model(inputs=[self.model.layers[0].input], outputs=[output_tensors[1]])
							else:
								raise ValueError("Pretrained Model's output tensors doesn't match number of classes in last layer.")

						self.new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
				else:
					output_tensors = self.model.outputs
					if len(output_tensors) == 1:
						if output_tensors[0].shape[1] == 1000:
							self.setup_to_transfer_learn()
							self.add_new_last_layer()
						else:
							if output_tensors[0].shape[1] == self.nb_classes:
								self.setup_to_transfer_learn()
								tensor_name = output_tensors[0].name
								if type(self.model.get_layer(tensor_name.split('/')[0])).__name__ != 'Reshape':
									self.model.get_layer(tensor_name.split('/')[0]).trainable = True
								else:
									self.model.get_layer(tensor_name.split('/')[0]).trainable = True
									last_second_layer = self.model.get_layer(tensor_name.split('/')[0]).input.name.split('/')[0]
									self.model.get_layer(last_second_layer).trainable = True
									last_third_layer = self.model.get_layer(last_second_layer).input.name.split('/')[0]
									self.model.get_layer(last_third_layer).trainable = True
								self.new_model = self.model
								self.new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
							else:
								self.setup_to_transfer_learn()
								self.add_new_last_layer()
					else:
						if len(output_tensors) != 2:
							raise ValueError("Pretrained Model should have only two output tensors. "
												"One for ImageNet output, Second for new classes.")

						if 'new_' in output_tensors[0].name:
							if output_tensors[0].shape[1] == self.nb_classes:
								self.setup_to_transfer_learn()
								tensor_name = output_tensors[0].name
								if type(self.model.get_layer(tensor_name.split('/')[0])).__name__ != 'Reshape':
									self.model.get_layer(tensor_name.split('/')[0]).trainable = True
								else:
									self.model.get_layer(tensor_name.split('/')[0]).trainable = True
									last_second_layer = self.model.get_layer(tensor_name.split('/')[0]).input.name.split('/')[0]
									self.model.get_layer(last_second_layer).trainable = True
									last_third_layer = self.model.get_layer(last_second_layer).input.name.split('/')[0]
									self.model.get_layer(last_third_layer).trainable = True
								self.new_model = Model(inputs=[self.model.layers[0].input], outputs=[output_tensors[0]])
								self.model = Model(inputs=[self.model.layers[0].input], outputs=[output_tensors[1]])
							else:
								raise ValueError("Pretrained Model's output tensors doesn't match number of classes in last layer.")
						else:
							if output_tensors[1].shape[1] == self.nb_classes:
								self.setup_to_transfer_learn()
								tensor_name = output_tensors[1].name
								if type(self.model.get_layer(tensor_name.split('/')[0])).__name__ != 'Reshape':
									self.model.get_layer(tensor_name.split('/')[0]).trainable = True
								else:
									self.model.get_layer(tensor_name.split('/')[0]).trainable = True
									last_second_layer = self.model.get_layer(tensor_name.split('/')[0]).input.name.split('/')[0]
									self.model.get_layer(last_second_layer).trainable = True
									last_third_layer = self.model.get_layer(last_second_layer).input.name.split('/')[0]
									self.model.get_layer(last_third_layer).trainable = True
								self.new_model = Model(inputs=[self.model.layers[0].input], outputs=[output_tensors[1]])
								self.model = Model(inputs=[self.model.layers[0].input], outputs=[output_tensors[0]])
							else:
								raise ValueError("Pretrained Model's output tensors doesn't match number of classes in last layer.")

						self.new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

			if self.training_mode == 2:
				if self.preserve_imagenet_classes:
					output_tensors = self.model.outputs
					if len(output_tensors) == 1:
						if self.model.outputs[0].shape[1] != 1000:
							raise ValueError("Pretrained Model doesn't have ImageNet output tensor."
												"Set: preserve_imagenet_classes = False")
						else:
							self.setup_to_finetune()
							self.add_new_last_layer()
					else:
						if len(output_tensors) != 2:
							raise ValueError("Pretrained Model should have only two output tensors. "
												"One for ImageNet output, Second for new classes.")

						if output_tensors[0].shape[1] == 1000:
							if output_tensors[1].shape[1] == self.nb_classes:
								self.setup_to_finetune()
								self.new_model = Model(inputs=[self.model.layers[0].input], outputs=[output_tensors[1]])
								self.model = Model(inputs=[self.model.layers[0].input], outputs=[output_tensors[0]])
							else:
								raise ValueError("Pretrained Model's output tensors doesn't match number of classes in last layer.")
						else:
							if output_tensors[0].shape[1] == self.nb_classes:
								self.setup_to_finetune()
								self.new_model = Model(inputs=[self.model.layers[0].input], outputs=[output_tensors[0]])
								self.model = Model(inputs=[self.model.layers[0].input], outputs=[output_tensors[1]])
							else:
								raise ValueError("Pretrained Model's output tensors doesn't match number of classes in last layer.")

						self.new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
				else:
					output_tensors = self.model.outputs
					if len(output_tensors) == 1:
						if output_tensors[0].shape[1] == 1000:
							self.setup_to_finetune()
							self.add_new_last_layer()
						else:
							if output_tensors[0].shape[1] == self.nb_classes:
								self.setup_to_finetune()
								self.new_model = self.model
								self.new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
							else:
								self.setup_to_finetune()
								self.add_new_last_layer()
					else:
						if len(output_tensors) != 2:
							raise ValueError("Pretrained Model should have only two output tensors. "
												"One for ImageNet output, Second for new classes.")

						if 'new_' in output_tensors[0].name:
							if output_tensors[0].shape[1] == self.nb_classes:
								self.setup_to_finetune()
								self.new_model = Model(inputs=[self.model.layers[0].input], outputs=[output_tensors[0]])
								self.model = Model(inputs=[self.model.layers[0].input], outputs=[output_tensors[1]])
							else:
								raise ValueError("Pretrained Model's output tensors doesn't match number of classes in last layer.")
						else:
							if output_tensors[1].shape[1] == self.nb_classes:
								self.setup_to_finetune()
								self.new_model = Model(inputs=[self.model.layers[0].input], outputs=[output_tensors[1]])
								self.model = Model(inputs=[self.model.layers[0].input], outputs=[output_tensors[0]])
							else:
								raise ValueError("Pretrained Model's output tensors doesn't match number of classes in last layer.")

						self.new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

			if self.training_mode == 3:
				self.setup_to_bottleneck()

			if self.training_mode == 4:
				if self.preserve_imagenet_classes:
					output_tensors = self.model.outputs
					if len(output_tensors) == 1:
						if output_tensors[0].shape[1] != 1000:
							raise ValueError("Pretrained Model doesn't have ImageNet output tensor."
												"Set: preserve_imagenet_classes = False")
						else:
							self.setup_to_scratch_learning()
							self.add_new_last_layer()
					else:
						if len(output_tensors) != 2:
							raise ValueError("Pretrained Model should have only two output tensors. "
												"One for ImageNet output, Second for new classes.")

						if output_tensors[0].shape[1] == 1000:
							if output_tensors[1].shape[1] == self.nb_classes:
								self.setup_to_scratch_learning()
								self.new_model = Model(inputs=[self.model.layers[0].input], outputs=[output_tensors[1]])
								self.model = Model(inputs=[self.model.layers[0].input], outputs=[output_tensors[0]])
							else:
								raise ValueError("Pretrained Model's output tensors doesn't match number of classes in last layer.")
						else:
							if output_tensors[0].shape[1] == self.nb_classes:
								self.setup_to_scratch_learning()
								self.new_model = Model(inputs=[self.model.layers[0].input], outputs=[output_tensors[0]])
								self.model = Model(inputs=[self.model.layers[0].input], outputs=[output_tensors[1]])
							else:
								raise ValueError("Pretrained Model's output tensors doesn't match number of classes in last layer.")

						self.new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
				else:
					output_tensors = self.model.outputs
					if len(output_tensors) == 1:
						if output_tensors[0].shape[1] == 1000:
							self.setup_to_scratch_learning()
							self.add_new_last_layer()
						else:
							if output_tensors[0].shape[1] == self.nb_classes:
								self.setup_to_scratch_learning()
								self.new_model = self.model
								self.new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
							else:
								self.setup_to_scratch_learning()
								self.add_new_last_layer()
					else:
						if len(output_tensors) != 2:
							raise ValueError("Pretrained Model should have only two output tensors. "
												"One for ImageNet output, Second for new classes.")

						if 'new_' in output_tensors[0].name:
							if output_tensors[0].shape[1] == self.nb_classes:
								self.setup_to_scratch_learning()
								self.new_model = Model(inputs=[self.model.layers[0].input], outputs=[output_tensors[0]])
								self.model = Model(inputs=[self.model.layers[0].input], outputs=[output_tensors[1]])
							else:
								raise ValueError("Pretrained Model's output tensors doesn't match number of classes in last layer.")
						else:
							if output_tensors[1].shape[1] == self.nb_classes:
								self.setup_to_scratch_learning()
								self.new_model = Model(inputs=[self.model.layers[0].input], outputs=[output_tensors[1]])
								self.model = Model(inputs=[self.model.layers[0].input], outputs=[output_tensors[0]])
							else:
								raise ValueError("Pretrained Model's output tensors doesn't match number of classes in last layer.")

						self.new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
		else:
			if self.training_mode == 1:
				self.setup_to_transfer_learn()
				self.add_new_last_layer()

			if self.training_mode == 2:
				self.setup_to_finetune()
				self.add_new_last_layer()

			if self.training_mode == 3:
				self.setup_to_bottleneck()

			if self.training_mode == 4:
				self.setup_to_scratch_learning()
				self.add_new_last_layer()

	def setup_to_transfer_learn(self):
		for layer in self.model.layers:
			layer.trainable = False

	def setup_to_scratch_learning(self):
		for layer in self.model.layers:
			layer.trainable = True

	def setup_to_bottleneck(self):
		note = "\nChoose the layer number from where you want to extract features to feed as input in SVM classifier for training: "
		self.select_bottleneck_layer(note)
		x = self.model.layers[self.bottleneck_layer_no - 1].output
		x = Flatten()(x)
		self.new_model = Model(inputs=[self.model.layers[0].input], outputs=[x])

	def setup_to_finetune(self):
		note = "\nChoose the layer number upto which you want to freeze the layers: "
		self.select_layer(note)
		for layer in self.model.layers[:self.freeze_layer_no]:
			layer.trainable = False
		for layer in self.model.layers[self.freeze_layer_no:]:
			layer.trainable = True

	def add_new_last_layer(self):
		if self.name == 'MobileNet':
			x = self.model.layers[-4].output
			x = Conv2D(self.nb_classes, (1, 1), padding='same', name='new_conv_preds')(x)
			x = Activation('softmax', name='new_act_softmax')(x)
			x = Reshape((self.nb_classes,), name='new_reshape_2')(x)
			self.new_model = Model(inputs=[self.model.layers[0].input], outputs=[x])
		elif self.name in self.model_dict.values():
			x = self.model.layers[-2].output
			for i in range(self.dense_layer):
				x = Dense(self.nb_classes, activation='softmax', name='new_prediction_'+str(i))(x)
				if self.dropout != 0.:
					x = Dropout(self.dropout, name='new_dense_dropout_'+str(i))(x)
			self.new_model = Model(inputs=[self.model.layers[0].input], outputs=[x])
		else:
			print('-'*120)
			print("NOTE: Make sure your pretrained model's last layer is prediction_layer. "
					"e.g., prediction_layer = Dense(nb_class, activation='Softmax')")
			print('-'*120, '\n')
			x = self.model.layers[-2].output
			for i in range(self.dense_layer):
				x = Dense(self.nb_classes, activation='softmax', name='new_prediction_'+str(i))(x)
				if self.dropout != 0.:
					x = Dropout(self.dropout, name='new_dense_dropout_'+str(i))(x)
			self.new_model = Model(inputs=[self.model.layers[0].input], outputs=[x])

		if self.training_mode == 1:
			optimizer = SGD(lr=1e-4, momentum=0.9)
		else:
			optimizer='adam'
		self.new_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

	def lr_schedule(self, epoch):
		lrate = self.initial_lrate * math.exp(-self.exp_drop*epoch)
		return lrate

	def create_callbacks(self):
		if self.is_colab:
			if self.preserve_imagenet_classes:
				modelCheckpoint_call = ckpt_callback.ModelCheckpoint(os.path.join(self.result_dir, 
																	'model.{epoch:02d}-{val_acc:.4f}.hdf5'), 
																	self.model, monitor='val_acc', verbose=1, save_best_only=False, 
																	save_weights_only=False, mode='auto', period=self.model_save_period)
			else:
				modelCheckpoint_call = ModelCheckpoint(os.path.join(self.result_dir, 
														'model.{epoch:02d}-{val_acc:.4f}.hdf5'), 
														monitor='val_acc', verbose=1, save_best_only=False, save_weights_only=False, 
														mode='auto', period=self.model_save_period)

			#tensorboard_call = TensorBoard(log_dir=os.path.join(self.result_dir, 'logs'), histogram_freq=0, batch_size=self.batch_size, write_graph=True, write_grads=True, write_images=True)
			if self.use_tensorboard:
				tbc = setup_tbc.TensorBoardColab(graph_path='log', startup_waiting_time=15)
				tensorboard_call = tbc_callback.TensorBoardColabCallback(tbc=tbc, batch_gene=self.validation_generator, 
																		nb_steps=self.nb_val_samples//self.batch_size, 
																		histogram_freq=self.histogram_freq, batch_size=self.batch_size, 
																		write_graph=True, write_grads=self.write_grads, 
																		write_images=self.write_images, embeddings_freq=0, 
																		embeddings_layer_names=None, embeddings_metadata=None, 
																		embeddings_data=None, update_freq='epoch')
		else:
			if self.preserve_imagenet_classes:
				modelCheckpoint_call = ckpt_callback.ModelCheckpoint('model.{epoch:02d}--{val_acc:.4f}.hdf5', 
																	monitor='val_acc', verbose=1, save_best_only=False, 
																	save_weights_only=False, mode='auto', period=self.model_save_period)
			else:
				modelCheckpoint_call = ModelCheckpoint('model.{epoch:02d}--{val_acc:.4f}.hdf5', 
														monitor='val_acc', verbose=1, save_best_only=False, save_weights_only=False, 
														mode='auto', period=self.model_save_period)

			if self.use_tensorboard:
				tensorboard_call = hist_callback.TensorBoardWrapper(batch_gene=self.validation_generator, 
																	nb_steps=self.nb_val_samples//self.batch_size, 
																	log_dir='./logs', histogram_freq=self.histogram_freq, 
																	batch_size=self.batch_size, write_graph=True, 
																	write_grads=self.write_grads, write_images=self.write_images, 
																	embeddings_freq=0, embeddings_layer_names=None, 
																	embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

		lrDecay_call = LearningRateScheduler(self.lr_schedule, verbose=1)

		if self.training_mode == 1:
			if self.use_tensorboard:
				self.callbacks = [modelCheckpoint_call, tensorboard_call]
			else:
				self.callbacks = [modelCheckpoint_call]

		if self.training_mode == 2:
			if self.use_tensorboard:
				self.callbacks = [modelCheckpoint_call, lrDecay_call, tensorboard_call]
			else:
				self.callbacks = [modelCheckpoint_call, lrDecay_call]

		if self.training_mode == 4:
			if self.use_tensorboard:
				self.callbacks = [modelCheckpoint_call, lrDecay_call, tensorboard_call]
			else:
				self.callbacks = [modelCheckpoint_call, lrDecay_call]

	def train(self):
		self.create_callbacks()

		if self.training_mode != 3:
			self.history_tl = self.new_model.fit_generator(
											self.train_generator,
											steps_per_epoch=self.nb_train_samples//self.batch_size,
											epochs=self.nb_epoch,
											validation_data=self.validation_generator,
											validation_steps=self.nb_val_samples//self.batch_size,
											class_weight='auto',
											callbacks=self.callbacks,
											max_queue_size=10, workers=self.cpu_workers, use_multiprocessing=False, shuffle=True,
											initial_epoch=self.initial_epoch
											)

			if self.preserve_imagenet_classes:
				self.new_model = Model(inputs=[self.model.layers[0].input], 
										outputs=[self.new_model.layers[-1].output, self.model.layers[-1].output])

				self.new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

				if self.is_colab:
					self.new_model.save(os.path.join(self.result_dir, 'final_' + self.name + '.hdf5'))
				else:
					self.new_model.save('final_' + self.name + '.hdf5')
			else:
				if self.is_colab:
					self.new_model.save(os.path.join(self.result_dir, 'final_' + self.name + '.hdf5'))
				else:
					self.new_model.save('final_' + self.name + '.hdf5')

			if self.is_colab and self.use_tensorboard:
				shutil.copytree('log', os.path.join(self.result_dir, 'tensorboard log'))
		else:
			self.train_svm()

	def train_svm(self):
		no = 0
		steps = self.nb_train_samples//self.batch_size

		X = []
		Y = []
		for x, y in self.train_generator:
			if no < steps:
				X.extend(self.new_model.predict(x))
				Y.extend(np.argmax(y, axis=1))
			else:
				break
			no += 1

		self.X = np.array(X)
		self.Y = np.array(Y)
		if self.take_y_n("Do you want to train SVM on extracted features [Y/N]: "):
			self.clf = SVC()
			self.clf.fit(self.X, self.Y)

	def predict(self, path):
		if self.is_colab:
			if os.path.exists(os.path.join('gdrive/My Drive', path)):
				path = os.path.join('gdrive/My Drive', path)
			elif os.path.exists(path):
				path = path
			else:
				raise ValueError("Path doesn't exist in Google-Drive/Colab-workspace @: ", path)
		else:
			if os.path.exists(path):
				path = path
			else:
				raise ValueError("Path doesn't exist @: ", path)

		if self.train_mode:
			if self.training_mode != 3:
				self.results = {}
				if os.path.isdir(path):
					for file in os.listdir(path):
						filepath = os.path.join(path, file)
						if self.preserve_imagenet_classes:
							new_class, old_class = self.get_prediction_result(filepath)
							self.results[filepath] = [new_class, old_class]
						else:
							new_class = self.get_prediction_result(filepath)
							self.results[filepath] = [new_class]
				else:
					if self.preserve_imagenet_classes:
						new_class, old_class = self.get_prediction_result(path)
						self.results[path] = [new_class, old_class]
					else:
						new_class = self.get_prediction_result(path)
						self.results[path] = [new_class]

				if self.is_colab:
					self.pd_result = write_result.show(self.results, self.result_dir, self.preserve_imagenet_classes)
			else:
				self.results = {}
				if os.path.isdir(path):
					for file in os.listdir(path):
						filepath = os.path.join(path, file)
						self.preserve_imagenet_classes = False
						x = self.new_model.predict(self.get_img_matrix(filepath))
						new_class = self.clf.predict(x)
						self.results[filepath] = [self.inv_class_mapping[new_class[0]]]
				else:
					self.preserve_imagenet_classes = False
					x = self.new_model.predict(self.get_img_matrix(path))
					new_class = self.clf.predict(x)
					self.results[path] = [self.inv_class_mapping[new_class[0]]]

				if self.is_colab:
					self.pd_result = write_result.show(self.results, self.result_dir, self.preserve_imagenet_classes)
		else:
			output_tensors = self.model.outputs
			self.results = {}
			if os.path.isdir(path):
				if self.class_mapping == None:
					if len(output_tensors) == 1:
						if output_tensors[0].shape[1] != 1000:
							raise ValueError("class_mapping is not provided for the model.")
						else:
							self.preserve_imagenet_classes = False
							for file in os.listdir(path):
								filepath = os.path.join(path, file)
								x = self.get_img_matrix(filepath)
								preds = self.model.predict(x)
								self.results[filepath] = [self.decode_fun(preds, top=1)[0][0][1]]
					else:
						raise ValueError("class_mapping is not provided for the model.")
				else:
					if type(self.class_mapping) == str:
						with open(self.class_mapping, 'r') as fp:
							self.class_mapping = json.load(fp)
					elif type(self.class_mapping) != dict:
						raise ValueError("class_mapping must be filepath to json or dict-mapping for class to index.")
					self.inv_class_mapping = {v: k for k, v in self.class_mapping.items()}

					if len(output_tensors) == 1:
						if len(self.class_mapping.keys()) != output_tensors[0].shape[1]:
							raise ValueError("class_mapping doesn't match output tensor size.")
						else:
							self.preserve_imagenet_classes = False
							for file in os.listdir(path):
								filepath = os.path.join(path, file)
								x = self.get_img_matrix(filepath)
								preds = self.model.predict(x)
								self.results[filepath] = [self.inv_class_mapping[np.argmax(preds)]]
					else:
						if len(output_tensors) != 2:
							raise ValueError("Model should have two output tensors. One for ImageNet classes and Second for your data")

						if output_tensors[0].shape[1] == 1000:
							if output_tensors[1].shape[1] != len(self.class_mapping.keys()):
								raise ValueError("class_mapping doesn't match output tensor size.")
							else:
								self.preserve_imagenet_classes = True
								for file in os.listdir(path):
									filepath = os.path.join(path, file)
									x = self.get_img_matrix(filepath)
									preds1, preds2 = self.model.predict(x)
									self.results[filepath] = [self.inv_class_mapping[np.argmax(preds2)], self.decode_fun(preds1, top=1)[0][0][1]]
						elif output_tensors[1].shape[1] == 1000:
							if output_tensors[0].shape[1] != len(self.class_mapping.keys()):
								raise ValueError("class_mapping doesn't match output tensor size.")
							else:
								self.preserve_imagenet_classes = True
								for file in os.listdir(path):
									filepath = os.path.join(path, file)
									x = self.get_img_matrix(filepath)
									preds1, preds2 = self.model.predict(x)
									self.results[filepath] = [self.inv_class_mapping[np.argmax(preds1)], self.decode_fun(preds2, top=1)[0][0][1]]
						else:
							raise ValueError("Expecting either of output tensor for ImageNet classes.")
			else:
				if self.class_mapping == None:
					if len(output_tensors) == 1:
						if output_tensors[0].shape[1] != 1000:
							raise ValueError("class_mapping is not provided for the model.")
						else:
							self.preserve_imagenet_classes = False
							x = self.get_img_matrix(path)
							preds = self.model.predict(x)
							self.results[path] = [self.decode_fun(preds, top=1)[0][0][1]]
					else:
						raise ValueError("class_mapping is not provided for the model.")
				else:
					if type(self.class_mapping) == str:
						with open(self.class_mapping, 'r') as fp:
							self.class_mapping = json.load(fp)
					elif type(self.class_mapping) != dict:
						raise ValueError("class_mapping must be filepath to json or dict-mapping for class to index.")
					self.inv_class_mapping = {v: k for k, v in self.class_mapping.items()}

					if len(output_tensors) == 1:
						if len(self.class_mapping.keys()) != output_tensors[0].shape[1]:
							raise ValueError("class_mapping doesn't match output tensor size.")
						else:
							self.preserve_imagenet_classes = False
							x = self.get_img_matrix(path)
							preds = self.model.predict(x)
							self.results[path] = [self.inv_class_mapping[np.argmax(preds)]]
					else:
						if len(output_tensors) != 2:
							raise ValueError("Model should have two output tensors. One for ImageNet classes and Second for your data")

						if output_tensors[0].shape[1] == 1000:
							if output_tensors[1].shape[1] != len(self.class_mapping.keys()):
								raise ValueError("class_mapping doesn't match output tensor size.")
							else:
								self.preserve_imagenet_classes = True
								x = self.get_img_matrix(path)
								preds1, preds2 = self.model.predict(x)
								self.results[path] = [self.inv_class_mapping[np.argmax(preds2)], self.decode_fun(preds1, top=1)[0][0][1]]
						elif output_tensors[1].shape[1] == 1000:
							if output_tensors[0].shape[1] != len(self.class_mapping.keys()):
								raise ValueError("class_mapping doesn't match output tensor size.")
							else:
								self.preserve_imagenet_classes = True
								x = self.get_img_matrix(path)
								preds1, preds2 = self.model.predict(x)
								self.results[path] = [self.inv_class_mapping[np.argmax(preds1)], self.decode_fun(preds2, top=1)[0][0][1]]
						else:
							raise ValueError("Expecting either of output tensor for ImageNet classes.")
			if self.is_colab:
				self.pd_result = write_result.show(self.results, '', self.preserve_imagenet_classes)

	def get_img_matrix(self, filepath):
		img = image.load_img(filepath, target_size=self.target_size)
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = self.preprocess_fun(x)
		return x

	def get_prediction_result(self, filepath):
		x = self.get_img_matrix(filepath)
		if self.preserve_imagenet_classes:
			new_preds, old_preds = self.new_model.predict(x)
			return self.inv_class_mapping[np.argmax(new_preds)], self.decode_fun(old_preds, top=1)[0][0][1]
		else:
			preds = self.new_model.predict(x)
			return self.inv_class_mapping[np.argmax(preds)]

	def select_bottleneck_layer(self, note):
		try:
			for i, layer in enumerate(self.model.layers):
				print(i + 1, layer.name, layer.input_shape)
			self.bottleneck_layer_no = int(input(note))
			if not (self.bottleneck_layer_no >= 1 and self.bottleneck_layer_no <= len(self.model.layers)):
				self.select_bottleneck_layer(note)
		except:
			print("\nChoose the right layer")
			self.select_bottleneck_layer(note)

	def select_training_mode(self, note):
		try:
			print('-'*80)
			print('1. Train only top of the model')
			print('2. Finetune the model upto top few layers')
			print('3. SVM Classifier from particular layer activations')
			print('4. Scratch full model training')
			print('-'*80, '\n')
			self.training_mode = int(input(note))
			if not (self.training_mode >= 1 and self.training_mode <= 4):
				self.select_training_mode(note)
		except:
			print('\nChoose right number :)')
			self.select_training_mode(note)

	def select_layer(self, note):
		try:
			for i, layer in enumerate(self.model.layers):
				print(i + 1, layer.name, layer.input_shape)
			self.freeze_layer_no = int(input(note))
			if not (self.freeze_layer_no >= 1 and self.freeze_layer_no <= len(self.model.layers)):
				self.select_layer(note)
		except:
			print("\nChoose the right layer")
			self.select_layer(note)

	def take_y_n(self, note):
		try:
			ans = input(note).lower()
			if ans == 'y':
				return True
			elif ans == 'n':
				return False
			else:
				self.take_y_n(note)
		except:
			self.take_y_n(note)

	def select_model(self, note):
		try:
			print('-'*80)
			print('1. Xception')
			print('2. VGG16')
			print('3. VGG19')
			print('4. ResNet50')
			print('5. InceptionV3')
			print('6. InceptionResNetV2')
			print('7. MobileNet')
			print('8. MobileNetV2')
			print('9. DenseNet121')
			print('10. DenseNet169')
			print('11. DenseNet201')
			print('12. NASNetMobile')
			print('13. NASNetLarge')
			print('-'*80, '\n')
			self.model_no = int(input(note))
			if not (self.model_no >= 1 and self.model_no <= 13):
				self.select_model(note)
		except:
			print('\nChoose right number :)')
			self.select_model(note)

	def set_model(self):
		if self.model_dict[self.model_no] == 'Xception':
			K.clear_session()
			self.model = xception.Xception(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
			from keras.applications.xception import preprocess_input, decode_predictions
			self.preprocess_fun = preprocess_input
			self.decode_fun = decode_predictions
			self.target_size = (299, 299)

		if self.model_dict[self.model_no] == 'VGG16':
			K.clear_session()
			self.model = vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
			from keras.applications.vgg16 import preprocess_input, decode_predictions
			self.preprocess_fun = preprocess_input
			self.decode_fun = decode_predictions
			self.target_size = (224, 224)

		if self.model_dict[self.model_no] == 'VGG19':
			K.clear_session()
			self.model = vgg19.VGG19(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
			from keras.applications.vgg19 import preprocess_input, decode_predictions
			self.preprocess_fun = preprocess_input
			self.decode_fun = decode_predictions
			self.target_size = (224, 224)

		if self.model_dict[self.model_no] == 'ResNet50':
			K.clear_session()
			self.model = resnet50.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
			from keras.applications.resnet50 import preprocess_input, decode_predictions
			self.preprocess_fun = preprocess_input
			self.decode_fun = decode_predictions
			self.target_size = (224, 224)

		if self.model_dict[self.model_no] == 'InceptionV3':
			K.clear_session()
			self.model = inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
			from keras.applications.inception_v3 import preprocess_input, decode_predictions
			self.preprocess_fun = preprocess_input
			self.decode_fun = decode_predictions
			self.target_size = (299, 299)

		if self.model_dict[self.model_no] == 'InceptionResNetV2':
			K.clear_session()
			self.model = inception_resnet_v2.InceptionResNetV2(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
			from keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
			self.preprocess_fun = preprocess_input
			self.decode_fun = decode_predictions
			self.target_size = (299, 299)

		if self.model_dict[self.model_no] == 'MobileNet':
			K.clear_session()
			self.model = mobilenet.MobileNet(input_shape=None, alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
			from keras.applications.mobilenet import preprocess_input, decode_predictions
			self.preprocess_fun = preprocess_input
			self.decode_fun = decode_predictions
			self.target_size = (224, 224)

		if self.model_dict[self.model_no] == 'MobileNetV2':
			K.clear_session()
			self.model = mobilenet_v2.MobileNetV2(input_shape=None, alpha=1.0, depth_multiplier=1, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000) 
			from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
			self.preprocess_fun = preprocess_input
			self.decode_fun = decode_predictions
			self.target_size = (224, 224)

		if self.model_dict[self.model_no] == 'DenseNet121':
			K.clear_session()
			self.model = densenet.DenseNet121(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
			from keras.applications.densenet import preprocess_input, decode_predictions
			self.preprocess_fun = preprocess_input
			self.decode_fun = decode_predictions
			self.target_size = (224, 224)

		if self.model_dict[self.model_no] == 'DenseNet169':
			K.clear_session()
			self.model = densenet.DenseNet169(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
			from keras.applications.densenet import preprocess_input, decode_predictions
			self.preprocess_fun = preprocess_input
			self.decode_fun = decode_predictions
			self.target_size = (224, 224)

		if self.model_dict[self.model_no] == 'DenseNet201':
			K.clear_session()
			self.model = densenet.DenseNet201(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
			from keras.applications.densenet import preprocess_input, decode_predictions
			self.preprocess_fun = preprocess_input
			self.decode_fun = decode_predictions
			self.target_size = (224, 224)

		if self.model_dict[self.model_no] == 'NASNetMobile':
			K.clear_session()
			self.model = nasnet.NASNetMobile(input_shape=None, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
			from keras.applications.nasnet import preprocess_input, decode_predictions
			self.preprocess_fun = preprocess_input
			self.decode_fun = decode_predictions
			self.target_size = (224, 224)

		if self.model_dict[self.model_no] == 'NASNetLarge':
			K.clear_session()
			self.model = nasnet.NASNetLarge(input_shape=None, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
			from keras.applications.nasnet import preprocess_input, decode_predictions
			self.preprocess_fun = preprocess_input
			self.decode_fun = decode_predictions
			self.target_size = (331, 331)

def main():
	#Retrain(full_data_dir_name='/home/dell/Desktop/Food image data')
	convnet = Retrain(train_dir_name='/home/dell/Desktop/Food image data/train_data', 
					val_dir_name='/home/dell/Desktop/Food image data/val_data')

if __name__ == "__main__":
	main()