# EEE-482_Computational_Neuroscience
Bilkent University, EEE-482 Computational Neuroscience course

# The project

The project uses three different feature selection methods and three different classification methods for solving the problem of visual object recognition. Pipelines investigate FMRI images of human subjects’ brain activity corresponding to eight different types of objects (face, cat, house, bottle, shoes, chair, scissors and nonsense pictures). For feature selection of the data, Ventral Temporal Cortex Masking, Analysis of Variance (ANOVA) and Recursive Feature Elimination (RFE) are used. Then, methods Support Vector Classifier (SVM), Logistic Regression (LR) and Linear Discriminant Analysis (LDA) are applied for classifiers. All of the classifiers are identified as successful for classifying FMRI images with proper regularization. Also, SVD feature reduction is used for the LDA classifier. To compare and understand the accuracy of feature selections and classifiers, confusion matrices are computed. These matrices along with accuracy scores of the classifier pipelines showed that a similar prediction accuracy can be achieved using RFE without using brain masks.
