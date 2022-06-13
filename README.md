# SMMA - Sparse MultiModal Analytics for Few Shot Learning


# Project Objective 
TEM as a powerful tool conveys atomic-level information of materials, which brings new interest of methodology development of data analysis in the materials science context. Moreover, the advanced technology extracts pixel-wise information of multiple dimensions at the same time, such as grey scale intensities and x-ray spectrums at the same time.

With the advantage of small learning samples, few-shot analysis compensates for the limited  sample size of TEM while delivering sufficient accuracy of learning results. 

In this project, we aim to combine the sparce analytics with the multimodality feature of TEM data for fast and accurate image characterization. As a result, the successfully built-up architecture would be the solution for the analysis of materials that have fleeting features. This would allow quicker discoveries of materials that could revolutionize clean tech, quantum computing, and any hardware based fields.

We have included our python notebook, detailing loading the pixels from the TEM images into matrices and dictionaries, which can then be used in the Few Shot Model. 

# Data Preprocessing Methodologies 
See and gfollow our TEM data python notebook. 
1. Manually crop the image by the domain boundaries.
2. Load brightness and x-ray signals of cropped domains.
3. Rearrange into matrices and dictionaries.

# Few-shot learning classification model

Our model implementation is based on prototypical networks introduced by Snell et al. in 2017 (Prototypical Networks for Few-shot Learning: https://arxiv.org/abs/1703.05175). The implementation is based on the github repo: https://github.com/cnielly/prototypical-networks-omniglot. 

From this original paper, the main idea behind this approach is the existence of an embedding in which points cluster around a single prototype representation for each class. In order words, we strive to find this embedding representations of our specific inputs such that the inputs that belong to the same class will have embeddings that are closer to each other; the embeddings of inputs that do not belong to the same class would be far away from each other. In this project, we learn neural network-based embeddings for processed input vectors using few-shot learning framework. 

The implementation, from input vector construction, few-shot learning model to training and testing, is all included in the Ipython notebook Few-shot Learning model.ipynb and can be downloaded to run in local CPU.  


# Download Git Repository
```
git clone git@github.com:SMMA-PNNL-2022/SMMA.git
```

# Create Environment
```
conda create -n hspy_environment
conda activate hspy_environment 
```

# Install Required Packages
- Hyperspy
- Matplotlib 
- Pytorch 
- Glob 
- Numpy

# Contributors
UW DIRECT Capstone Group Members - Yifei He, Jiayi Li, Ryan Littrell
Supervisors and Mentors: Steven R. Spurgeon, Sarah Akers, Christina Doty
