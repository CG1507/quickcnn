import setuptools

with open("README.md", "r") as fh:
	long_description = fh.read()

setuptools.setup(
	name="quickcnn",
	version="0.0.5",
	author="Ghanshyam Chodavadiya",
	author_email="g8ghanshym@gmail.com",
	description="Library for ConvNet training on colab",
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
