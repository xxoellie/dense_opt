import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dense_optimization",
    version="0.0.1",
    author="Dense Team",
    author_email="",
    description="A python package to help fold BatchNormalization on Sequential Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    project_urls={"": "",},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: No License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "dense_optimization"},
    packages=setuptools.find_packages(where="dense_optimization"),
    python_requires=">=3.7",
)