import setuptools

# with open("README.md", "r") as fh:
#     long_description = fh.read()
long_description = ' '

setuptools.setup(
    name="bnbML-ArtistBanda",
    version="0.0.1",
    author="Chirag Gomber",
    author_email="chirag.codes@gmail.com",
    description="Python implementations of some of the fundamental Machine Learning models and algorithms from scratch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ArtistBanda/Bread-and-Butter_Machine-Learning",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    license='LICENSE'
)
