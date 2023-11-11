from setuptools import setup, find_packages
import codecs
import os

VERSION = "0.0.1"
DESCRIPTION = "Face detection library with privacy controls: Blurring, Exclusion, and Specific Face detection"

# Setting up
setup(
    name="occultus",
    version=VERSION,
    author="Mark Avila",
    author_email="<markavila.dev@gmail.com>",
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=["opencv-python", "imread-from-url", "onnxruntime-gpu"],
    keywords=[
        "python",
        "object detection",
        "face detection",
        "face recognition",
        "camera stream",
    ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
)
