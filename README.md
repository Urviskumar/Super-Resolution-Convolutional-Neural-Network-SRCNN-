# Super-Resolution Convolutional Neural Network (SRCNN):

## Basic Overview

 <p align="justify">This project demonstrated the effective use of computer
 vision techniques, control algorithms, and the YOLOv5 model
in developing an autonomous navigation system. Despite the
 challenges faced, the project was successful in achieving
 its objectives and contributed to the field of autonomous
 navigation.</p>

## A. Introduction
<p align="justify">The Super-Resolution Convolutional Neural Network (SRCNN) is a pioneering deep learning model developed for single-image super-resolution. It was proposed by Chao Dong, Chen Change Loy, Kaiming He, and Xiaoou Tang in their paper titled "Image Super-Resolution Using Deep Convolutional Networks" in 2014.</p>

## Key components:<p align="justify">
1. **Patch Extraction and Representation**: Initially, low-resolution image patches are extracted and represented as high-dimensional feature vectors.
2. Non-linear Mapping**: These feature vectors undergo non-linear mapping through a convolutional neural network (CNN). This CNN consists of multiple layers of convolution, each followed by a ReLU activation function.
3. Reconstruction: The mapped high-dimensional feature vectors are reconstructed to obtain the high-resolution image.</p>

## Advantages:<p align="justify">
- SRCNN was one of the early successful attempts at utilizing deep learning for single-image super-resolution.<br>
- It showed significant improvements over traditional interpolation-based methods in terms of perceptual quality metrics like Peak Signal-to-Noise Ratio (PSNR) and Mean Squared Error (MSE).<br><br>

Single image super-resolution, which aims at
 recovering a high-resolution image from a single low
resolution image, is a classical problem in computer
 vision. This problem is inherently ill-posed since a mul
tiplicity of solutions exist for any given low-resolution
 pixel. In other words, it is an underdetermined in
verse problem, of which solution is not unique. Such
 a problem is typically mitigated by constraining the
 solution space by strong prior information.
Image super-resolution is the process of generating a high-resolution (HR) image from a low-resolution (LR) input. This problem is crucial in various fields such as computer vision, medical imaging, and satellite imagery analysis. The importance lies in enhancing the visual quality of images, enabling better perception and analysis of details. Traditional methods for image super-resolution often suffer from computational complexity and the loss of image details.</p>
<img src="https://pic2.zhimg.com/v2-48339af4c2ac2ad7f858eecf513dfacd_r.jpg" alt="SRCNN Model Figure" width="120%">

The Super-Resolution Convolutional Neural Network (**SRCNN**) method addresses these **challenges** by leveraging deep learning techniques. SRCNN has shown promising results in generating high-quality super-resolved images efficiently. It builds upon prior work in **deep learning-based** image processing, particularly in convolutional neural networks (CNNs) and image reconstruction.

## Methodology:<p align="justify">
In this project, we show that the aforementioned pipeline is equivalent to a deep convolutional neural network. Motivated by this fact, we consider a convolutional neural network that directly learns an end-to-end mapping between low- and high-resolution images. This method differs fundamentally from existing external example-based approaches, in that this method does not explicitly learn the dictionaries  or manifolds  for modelling the patch space. These are implicitly achieved via hidden layers. Furthermore, the patch extraction and aggregation are also formulated as convolutional layers, so are involved in the optimization. In this method, the entire SR pipeline is fully obtained through learning, with little pre/postprocessing. The proposed SRCNN has several appealing properties. First, its structure is intentionally designed with simplicity in mind, and yet provides superior accuracy compared with state-of-the-art example-based methods.</p>
