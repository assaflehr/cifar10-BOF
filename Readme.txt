CIFAR10 without NN
=======================
I first started with a NN, but I saw that they achieve above 95% accuracy out-of-box on CIFAR10, without any new code on my part. So I decided instead to use the method we learned in class, where I can do some modifications to the initial method.
This method does not get more than 80%  (but it was the state-of-the-art in those days...)

The High-level-idea: 
1. Find features  (with detectors like SIFT , or using dense-grid )
2. Use KMeans to find the common features
3. Use Bag-of-Words apprach to classificaiton.
(and a lot of interesting tweaks inside, like ZCA-whitening, soft represetion, coarse spatial pooling)

Relevant articles:

* An Analysis of Single-Layer Networks in Unsupervised Feature Learning / Adam Coates, Honglak Lee, Andrew Y. Ng.
* Learning Feature Representations with K-means/ Adam Coates, Andrew Y. Ng.
Zero Component Analysis (ZCA) is explained in Appendix A of https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf



CIFAR10 CLASSIFCATION 
=============================
1. Before running, please copy cifar10 into the data folder
data\data_batch_1..5
data\test_batch


How to use this notebook
I recommend looking at the code and the visualizations in the notebook version, but to re-run experiments using command-line argument, please use the .py version of this notebook

How to run standalone? <code>
python cifar10-kmeans-patches-dev.py --help                                # help for all the options
python cifar10-kmeans-patches-dev.py --test                                # for test on pkl results
python cifar10-kmeans-patches-dev.py --train --K 100 --subset 5000  --aug  # for training with different parameters 


data/ should contains cifar10 dataset
pkl/ should contain pkl files of the codebook, whitener and svm file, with the test-score inside the file name



