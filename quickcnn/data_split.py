import os
import shutil
import random

def create_new_dirs(directory, classes):
	if not os.path.exists(os.path.join(directory, 'train_data')):
		os.makedirs(os.path.join(directory, 'train_data'))

	if not os.path.exists(os.path.join(directory, 'val_data')):
		os.makedirs(os.path.join(directory, 'val_data'))

	for subdir in classes:
		if not os.path.exists(os.path.join(os.path.join(directory, 'train_data'), subdir)):
			os.makedirs(os.path.join(os.path.join(directory, 'train_data'), subdir))

		if not os.path.exists(os.path.join(os.path.join(directory, 'val_data'), subdir)):
			os.makedirs(os.path.join(os.path.join(directory, 'val_data'), subdir))

def move_data(directory, classes, samples_count, filepaths, fraction):
	for subdir in classes:
		train_samples = samples_count[subdir]*fraction//100
		files = []
		for file in filepaths[subdir]:
			files.append(os.path.join(os.path.join(directory, subdir), file))
		random.shuffle(files)
		train_files = files[:train_samples]
		val_files = files[train_samples:]

		for train_file in train_files:
			shutil.move(train_file, os.path.join(os.path.join(os.path.join(directory, 'train_data'), subdir), train_file.split('/')[-1]))

		for val_file in val_files:
			shutil.move(val_file, os.path.join(os.path.join(os.path.join(directory, 'val_data'), subdir), val_file.split('/')[-1]))
		print('Class: ', subdir, ' | data splitted |')

def split(directory, fraction):
	classes = []
	for subdir in sorted(os.listdir(directory)):
		if os.path.isdir(os.path.join(directory, subdir)):
			classes.append(subdir)

	samples_count = {}
	filepaths = {}

	for subdir in classes:
		filepaths[subdir] = os.listdir(os.path.join(directory, subdir))
		samples_count[subdir] = len(filepaths[subdir])

	print('-'*120)
	print('Total image_sample per class: (Before Data Split)')
	print(samples_count)
	print('-'*120)
	
	create_new_dirs(directory, classes)
	print()
	print('-'*120)
	move_data(directory, classes, samples_count, filepaths, fraction)
	print('-'*120)
	print()

	samples_count = {}
	filepaths = {}

	for subdir in classes:
		filepaths[subdir] = os.listdir(os.path.join(os.path.join(directory, 'train_data'), subdir))
		samples_count[subdir] = len(filepaths[subdir])
	print('-'*120)
	print('Total train image_sample per class: (After Data Split)')
	print(samples_count)
	print('-'*120)

	samples_count = {}
	filepaths = {}
	for subdir in classes:
		filepaths[subdir] = os.listdir(os.path.join(os.path.join(directory, 'val_data'), subdir))
		samples_count[subdir] = len(filepaths[subdir])

	print('-'*120)
	print('Total validation image_sample per class: (After Data Split)')
	print(samples_count)
	print('-'*120)

	for subdir in sorted(os.listdir(directory)):
		if os.path.isdir(os.path.join(directory, subdir)):
			if subdir != 'train_data' and subdir != 'val_data':
				shutil.rmtree(os.path.join(directory, subdir))

	return os.path.join(directory, 'train_data'), os.path.join(directory, 'val_data')

def main():
	directory = '/home/dell/Desktop/Food image data/'
	train_dir, val_dir = split(directory, 80)
	print(train_dir, val_dir)

if __name__ == "__main__":
	main()