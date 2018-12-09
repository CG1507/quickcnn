import tensorflow as tf
from keras.callbacks import TensorBoard
from . import hist_callback
import time
import os
import io

class TensorBoardColabCallback(hist_callback.TensorBoardWrapper):
	def __init__(self, tbc=None, batch_gene=None, nb_steps=1, histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, 
				write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, 
				update_freq='epoch', **kwargs):
		# Make the original `TensorBoard` log to a subdirectory 'training'

		if tbc is None:
			return

		log_dir = tbc.get_graph_path()

		training_log_dir = os.path.join(log_dir, 'training')
		super(TensorBoardColabCallback, self).__init__(batch_gene=batch_gene, nb_steps=nb_steps, log_dir=training_log_dir, 
														histogram_freq=histogram_freq, batch_size=batch_size, 
														write_graph=write_graph, write_grads=write_grads, write_images=write_images, 
														embeddings_freq=embeddings_freq, embeddings_layer_names=embeddings_layer_names, 
														embeddings_metadata=embeddings_metadata, embeddings_data=embeddings_data, 
														update_freq=update_freq, **kwargs)

		# Log the validation metrics to a separate subdirectory
		self.val_log_dir = os.path.join(log_dir, 'validation')

	def set_model(self, model):
		# Setup writer for validation metrics
		self.val_writer = tf.summary.FileWriter(self.val_log_dir)
		super(TensorBoardColabCallback, self).set_model(model)

	def on_epoch_end(self, epoch, logs=None):
		# Pop the validation logs and handle them separately with
		# `self.val_writer`. Also rename the keys so that they can
		# be plotted on the same figure with the training metrics
		logs = logs or {}
		val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}

		for name, value in val_logs.items():
			# print('val_logs:',epoch, name, value)
			summary = tf.Summary()
			summary_value = summary.value.add()
			summary_value.simple_value = value.item()
			summary_value.tag = name
			self.val_writer.add_summary(summary, epoch)
		self.val_writer.flush()

		# Pass the remaining logs to `TensorBoard.on_epoch_end`
		logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
		super(TensorBoardColabCallback, self).on_epoch_end(epoch, logs)

	def on_train_end(self, logs=None):
		super(TensorBoardColabCallback, self).on_train_end(logs)
		self.val_writer.close()
