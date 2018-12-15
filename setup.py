import setuptools

long_description="Main idea of QuickCNN is to train deep ConvNet without diving into architectural details. QuickCNN works as an interactive tool for transfer learning, finetuning, and scratch training with custom datasets. It has pretrained model zoo and also works with your custom keras model architecture."

setuptools.setup(
	name="quickcnn",
	version="0.0.12",
	author="Ghanshyam Chodavadiya",
	author_email="g8ghanshym@gmail.com",
	description="ConvNet architectures on Google-Colaboratory",
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="https://github.com/CG1507/quickcnn",
	packages=setuptools.find_packages(),
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
	],
)
