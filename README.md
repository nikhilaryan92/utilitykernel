# Proposal of Utility Kernel for Breast Cancer Survival Estimation

# References

Our manuscipt titled with "Utility Kernel for Breast Cancer Survival Estimation" has been submitted at IEEE/ACM Transactions on Computational Biology and Bioinformatics.

# Requirements
[python 3.9.7](https://www.python.org/downloads/)

[TensorFllow 2.4.1](https://www.tensorflow.org/install/)

[keras 2.4.3](https://pypi.org/project/Keras/)

[scikit-learn 1.0.2](http://scikit-learn.org/stable/)

[matplotlib 3.5.1](https://matplotlib.org/users/installing.html)



# Usage
utility_svm.py

-> Running the utility_svm.py will produce a file "output.csv" and "......._results.csv", where ..... stands for the modality name or combination of modalities. First row of the "output.csv" shows the true class labels of all instances followed by the predicted class labels from particular modalities for which utility SVM has been executed. "......._results.csv" shows the values of Area under ROC curves and tp, tn, fp and fn from particular modalities for which utility SVM has been executed.


