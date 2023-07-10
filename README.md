# GAN-based-Image-Inpainting-and-Super-Resolution-with-MAP-Estimation
\item Designed and implemented a Generative Adversarial Network (GAN)-based model for image Inpainting and Super-resolution tasks,
utilizing the Maximum a Posteriori (MAP) estimation technique. The model's generator was trained on MNIST dataset images of the digit 8.
It achieved remarkable results in both tasks.

Project developed from scratch during the Machine Learning Course (University of Patras).
To Run Gan_Inpainting.py:
1)pip istall numpy, scipy, matplotlib.
2) Put script and "data21.mat","data22.mat" in the same directory.
3) Run!

Description:
1) Load Generator Weights from data21.mat and distorted,ideal examples from data22.mat.
2) Prepare data: By changing N you select how many pixels to remove from Pic.
3) Relu: Define Relu function.
4) Relu_der: Define ReLu derivative.
5) sigmoid: Define Sigmoid function.
6) sigmoid_der: Define sigmoid derivative.
7) F: Define the MAP Loss.
8) F_der: Define the F derivative.
9) Main: Define parameters. Start the training loop in which: Do front propagation,
   calculate gradients, find cost, calculate ADAM, do the back propagation.
10) Plot images and loss to show the effectiveness of the algorithm.

To Run Gan_SuperRes.py:
1)pip istall numpy, scipy, matplotlib.
2) Put script and "data21.mat","data23.mat" in the same directory.
3) Run!

Description:
Majority of the code remains the same. The major difference is the creation the new transform matrix.
