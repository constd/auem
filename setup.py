import setuptools

setuptools.setup(
    name="auem",
    version="0.0.1",
    author="Konstantinos Dimitriou, Christopher Jacobi",
    author_email="const@embeddingspace.com",
    url="https://github.com/constd/auem",
    packages=setuptools.find_packages(),
    entry_points={"console_scripts": ["auem = auem.train:entry"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
