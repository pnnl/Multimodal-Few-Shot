SMMA - Sparse MultiModal Analytics for Few Shot Learning
=====


Project Objective
----
Transmission electron microscopy (TEM) is a powerful tool that probes atomic-level features of materials, including structure, chemistry, and defects. Because of its importance to materials science and chemistry, there is considerablye interest in the application of data science to classify and extract key features from microscopy data. Moreover, these instruments can generate a range of data simulatenously, motivating the need for multimodal analytics approaches combined with generalizable models suited to discovery cycles.

In this project, we aim to combine emerging few-shot machine learning (ML) with a multimodal framework to aid in fast and accurate microscope data quantification. We explore the necesssary architecture and data pre-processing required for such multimodal analysis, laying the groundwork for subsequent more complex models. We have included our python notebook, detailing loading the pixels from the TEM images into matrices and dictionaries, which can then be used in few-shot model.

This code is the result of a project for the University of Washington Data Intensive Research Enabling Clean Technologies (UW-DIRECT) capstone program. For more information on the program, visit: https://depts.washington.edu/uwdirect/

How to Use SMMA
====

Data Pre-Processing
----
See and follow our TEM data python notebook:

1. Manually crop the image by the domain boundaries.
2. Load brightness and x-ray signals of cropped domains.
3. Rearrange into matrices and dictionaries.

Few-shot Learning Classification Model
----

Our model implementation is based on prototypical networks introduced by Snell et al. in 2017 (Prototypical Networks for Few-shot Learning: https://arxiv.org/abs/1703.05175). The implementation is based on the github repo: https://github.com/cnielly/prototypical-networks-omniglot. Additional discussion of the application of few-shot learning to microscopy data is given in Akers et al. (https://doi.org/10.1038/s41524-021-00652-z]).

From this original paper, the main idea behind this approach is the existence of an embedding in which points cluster around a single prototype representation for each class. In order words, we strive to find this embedding representations of our specific inputs such that the inputs that belong to the same class will have embeddings that are closer to each other; the embeddings of inputs that do not belong to the same class would be far away from each other. In this project, we learn neural network-based embeddings for processed input vectors using few-shot learning framework. 

The implementation, from input vector construction, few-shot learning model to training and testing, is all included in the Ipython notebook Few-shot Learning model.ipynb and can be downloaded to run in local CPU.

Installation
====

Download Git Repository
----
```
git clone git@github.com:SMMA-PNNL-2022/SMMA.git
```

Create Environment
----
```
conda create -n hspy_environment
conda activate hspy_environment 
```

Install Required Packages
----
- Hyperspy
- Matplotlib 
- Pytorch 
- Glob 
- Numpy

Additional Reading
======

To learn more about the application of few-shot ML to electron microscopy data, please consult our following publications:

 • Doty, C., Gallagher, S., Cui, W., Chen, W., Bhushan, S., Oostrom, M., Akers, S., & Spurgeon, S. R. (2022). Design of a Graphical User Interface for Few-Shot Machine Learning Classification of Electron Microscopy Data. Computational Materials Science, 203, 111121. [https://doi.org/10.1016/j.commatsci.2021.111121]
 
 • Akers, S., Kautz, E., Trevino-Gavito, A., Olszta, M., Matthews, B. E., Wang, L., Du, Y., & Spurgeon, S. R. (2021). Rapid and flexible segmentation of electron microscopy data using few-shot machine learning. npj Computational Materials, 7(1), 187. [https://doi.org/10.1038/s41524-021-00652-z]

Contact Information
 ======================

 For questions, contact Steven Spurgeon (steven.spurgeon@pnnl.gov).

 Team Members
-------------

 University of Washington DIRECT Capstone Program Students: Yifei He, Jiayi Li, and Ryan Littrell

 Pacific Northwest National Laboratory: Christina Doty, Sarah Akers, and Steven R. Spurgeon

Acknowledgments
======================

C.D., S.A., and S.R.S. were supported by a Chemical Dynamics Initiative (CDi) Laboratory Directed Research and Development (LDRD) project at Pacific Northwest National Laboratory (PNNL). PNNL is a multiprogram national laboratory operated for the U.S. Department of Energy (DOE) by Battelle Memorial Institute under Contract No. DE-AC05-76RL0-1830. Experimental sample preparation was performed at the Environmental Molecular Sciences Laboratory (EMSL), a national scientific user facility sponsored by the Department of Energy's Office of Biological and Environmental Research and located at PNNL. TEM data was collected in the Radiological Microscopy Suite (RMS), located in the Radiochemical Processing Laboratory (RPL) at PNNL. Y.H., J.L., and R.L. acknowledge support from the University of Washington Clean Energy Institute and the National Science Foundation Research Traineeship under Award NSF DGE-1633216.

Usage License
======================
Copyright Battelle Memorial Institute

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. This material was prepared as an account of work sponsored by an agency of the United States Government. Neither the United States Government nor the United States Department of Energy, nor Battelle, nor any of their employees, nor any jurisdiction or organization that has cooperated in the development of these materials, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness or any information, apparatus, product, software, or process disclosed, or represents that its use would not infringe privately owned rights.

Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof, or Battelle Memorial Institute. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.

PACIFIC NORTHWEST NATIONAL LABORATORY operated by BATTELLE for the UNITED STATES DEPARTMENT OF ENERGY under Contract DE-AC05-76RL01830
