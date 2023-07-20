# GAN-based-Image-Inpainting-and-Super-Resolution-with-MAP-Estimation
Designed and implemented a Generative Adversarial Network (GAN)-based model for image Inpainting and Super-resolution tasks,
utilizing the Maximum a Posteriori (MAP) estimation technique. The model's generator was trained on MNIST dataset images of the digit 8.
It achieved remarkable results in both tasks.

Project developed from scratch during the Machine Learning Course (University of Patras).
To Run Gan_Inpainting.py:
1)pip istall numpy, scipy, matplotlib.
2) Put script and "data21.mat","data22.mat" in the same directory and change path in the Load_files.py.
3) Run!

Description:
1) Load_files.py: Load Generator Weights from data21.mat and distorted,ideal examples from data22.mat.
2) MAP_Fun.py: Representer Theorem functions.
3) ML_Fun.py: Sigmoid, ReLu and their derivatives.
4) TransMatrix: |Create the Transform matrix for SuperRes.
5) GAN_Inpainting.py, GAN_SuperRes.py: MAIN in which train the GAN using back propagation with ADAM optimization and converge to a result.
    Plot the original picture next to the distorted and final the generated one.

Thank you.
