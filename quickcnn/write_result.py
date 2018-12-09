import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd

def grid_display(list_of_images, list_of_titles=[], no_of_columns=2, figsize=(10,10)):
	font = {'family': 'serif',
		'color':  'black',
		'weight': 'bold',
		'size': 14,
		}
	fig = plt.figure(figsize=figsize)
	column = 0
	for i in range(len(list_of_images)):
		column += 1
		#  check for end of column and create a new figure
		if column == no_of_columns+1:
			fig = plt.figure(figsize=figsize)
			column = 1
		fig.add_subplot(1, no_of_columns, column)
		plt.imshow(list_of_images[i])
		plt.axis('off')
		if len(list_of_titles) >= len(list_of_images):
			plt.title(list_of_titles[i], fontdict=font)

def create_csv(results, result_dir, preserve_imagenet_classes):
	if preserve_imagenet_classes:
		with open(os.path.join(result_dir, 'result.csv'), 'w')as fp:
			fp.write('filepath,new_class,old_class\n')
			for file in results:
				fp.write(str(file) + ',' + str(results[file][0]) + ',' + str(results[file][1]) + '\n')
	else:
		with open(os.path.join(result_dir, 'result.csv'), 'w')as fp:
			fp.write('filepath,class\n')
			for file in results:
				fp.write(str(file) + ',' + str(results[file][0]) + '\n')

def show(results, result_dir, preserve_imagenet_classes):
	create_csv(results, result_dir, preserve_imagenet_classes)

	res = pd.read_csv(os.path.join(result_dir, 'result.csv'))

	images = []
	titles = []
	if preserve_imagenet_classes:
		for file in results:
			img = cv2.imread(file)
			images.append(img)
			titles.append("NEW CLASS: " + str(results[file][0]) + '\n' + "IMAGENET CLASS: " + str(results[file][1]))
	else:
		for file in results:
			img = cv2.imread(file)
			images.append(img)
			titles.append("CLASS: " + str(results[file][0]))
	images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]   
	grid_display(images, titles, 3, (20,20))	

	return res

def run():
	show(results, result_dir, preserve_imagenet_classes)

if __name__ == "__main__":
	run()
