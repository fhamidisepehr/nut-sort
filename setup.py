from setuptools import setup, find_packages


setup(
    name='nutsort',  # Name of the package
    version='0.1.0',  # Initial version of the package
    description='A simple package for sorting nuts',  # Short description of the package
    packages=find_packages(),  # Automatically find and include all packages in the directory
    install_requires=[  # List your package dependencies here
        "torch==2.2.2",
        "torchsummary==1.5.1",
        "torchvision==0.17.2",
        "numpy==1.26.4",
    ],
    python_requires='>=3.10',  # Specify the Python versions supported
)
