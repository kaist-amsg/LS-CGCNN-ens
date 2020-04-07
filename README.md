Labeled Site CGCNN with ensemble (LS-CGCNN-ens)
===========================================
The LS-CGCNN-ens is a python code for comprehensive deep learning of binding energy predictions developed by Jung group at KAIST, and Ulissi group at Carnegie Mellon University. The model leverages the concept of binding site, where the element of binding site atoms are given a pseudo-element. Please submit issues to this github page for any mistakes, or improvements. 

Developers
----------
Geun Ho Gu (ghgu@kaist.ac.kr) <- Current maintainer

Juhwan Noh (jhwan@kaist.ac.kr)

Dependencies
------------
-  Python3
-  Numpy
-  Pytorch
-  Pymatgen
-  Sklearn

Installation
------------
1. Clone this repository:
```
git clone https://github.com/kaist-amsg/LS-CGCNN-ens
```
2. Run some example to test:
```
python FindSite.py 
python Use.py
```
Above code find sites and label them using alpha shape for CO binding energy, and cgcnn is used to predict binding energy.

Guide
-----

FindSite.py is strictly for using the model where alpha shape is used to identify every possible binding sites. Use.py is used to predict binding energy.

For training the model, we included the CO binding energy training set in data/. Train.py can be run to reproduce the test errors in the manuscript. 

For converting the raw data the labeled site representation or training set in data/, see the MakeData.py in rawdata folder. We couldn't upload the raw data as well the training set for H binding energy due to the file size restriction. We can provide this upon request (ghgu@kaist.ac.kr).

   
Publications
------------
If you use this code, please cite:

Geun Ho Gu, Juhwan Noh, Sungwon Kim, Seoin Back, Zachary Ulissi, Yousung Jung. "Practical Deep-Learning Representation for Fast and Accurate Heterogeneous Catalyst Screening" *Journal of Physical Chemistry Letters* (in press) 


Additional Info
===============
This code base is a lightweight package that includes training (Train.py), test (Test.py), and usage (Use.py) code. Currently, the database used to train is not included. Please contact the Ulissi group to obtain the latest database. The example of data set format is in /example/ folder. You may notice that the intial batch is slow (for training, testing, using). This is due to the cost of finding neighbors in the first run. For multiple runs using the same data set, I would recommend pre-determining these neighbors (see rawdata/MakeData.py and PickleData in cgcnn/data.py) to speed up the training. 
