# NetRNApan
## Deciphering RNA modification and post-transcriptional regulation by deep learning framework 

RNA modification, which is evolutionarily conserved, is crucial for modulating various biological functions and disease pathogenesis. High resolution transcriptome-wide mapping of RNA modifications has facilitated both data resources and computational prediction of RNA modification. While these prediction algorithms are promising, they are limited in interpretability or generalizability, or the capacity for discovering novel post-transcriptional regulations. Here, we present NetRNApan, a deep learning framework for RNA modification site prediction, motif discovery and trans-regulatory factor identification. Using m5U profiles generated by FICC-seq and miCLIP-seq technologies as cases, we demonstrated the accuracy of NetRNApan with more efficient and interpretive feature representations. Decoding the informative characteristics detected by NetRNApan uncovered five representative clusters with consensus motifs that may be essential for m5U modification. Furthermore, NetRNApan revealed interesting trans-regulatory factors and provided a protein-binding perspective for investigating the function of RNA modifications. Specifically, we discovered 21 potential functional RNA-binding proteins (RBPs) whose binding sites were significantly linked to the extracted top-scoring motifs. Two examples are ANKHD1 and RBM4 with potential regulatory function of RNA modifications. Further analyses of identified RBPs revealed new insights into post-transcriptional regulatory mechanisms of m5U, such as gene expression, RNA splicing, and RNA transport. NetRNApan and the findings will be helpful for accurate and high-throughput detection of RNA modification sites in the study of mRNA regulation. NetRNApan is freely available at https://github.com/bsml320/NetRNApan.

<div align=center><img src="https://bioinfo.uth.edu/iapp/github/Figure1.jpg" width="800px"></div>

# Installation
Download NetRNApan by
```
git clone https://github.com/bsml320/NetRNApan
```
Installation has been tested in Linux server, CentOS Linux release 7.8.2003 (Core), with Python 3.7. Since the package is written in python 3x, python3x with the pip tool must be installed. NetRNApan uses the following dependencies: numpy, scipy, pandas, h5py, keras version=2.3.1, tensorflow=1.15 shutil, and pathlib. We highly recommend that users leave a message under the NetRNApan issue interface (https://github.com/bsml320/NetRNApan/issue) when encountering any installation and running problems. We will deal with it in time. You can install these packages by the following commands:
```
conda create -n NetRNApan python=3.7
conda activate NetRNApan
pip install pandas
pip install numpy
pip install scipy
pip install -v keras==2.3.1
pip install -v tensorflow==1.15
pip install seaborn
pip install shutil
pip install protobuf==3.20
pip install h5py==2.10.0
```
# Performance
To evaluate the prediction performance of NetBCE, the 5-fold CV was performed on the training dataset. The ROC curves were drawn and the corresponding AUC values were calculated. We found that NetBCE had high performance with the average AUC values of 0.8455 by 5-fold CV, with a range from 0.8379 to 0.8528. Since the number of epitopes and non-epitopes were not balanced in the training dataset, we also performed PR analysis and calculated the corresponding AUC values. The PR curve indicates the trade-off between the amount of false positive predictions compared to the amount of false negative predictions. NetBCE achieved average PR AUC values of 0.6165, suggesting our model had great potential in predicting functional epitopes the high precision. We compared the performance of NetBCE with other 6 ML-based methods regarding AUC value (AB, DT, GB, KNN, LR and RF). We observed that average AUC of NetBCE was 8.77-21.58% higher than those of the other six ML-based methods. In addition, we compared NetBCE with other existing tools based on the curated independent dataset. NetBCE had high performance with the AUC values of 0.8400 on the independent dataset, and achieved a ≥ 22.06% improvement of AUC value for the B cell epitope prediction compared to other tools

![image](https://github.com/BioDataStudy/NetBCE/blob/main/models/github_3.jpg)

# Interpretability
To elucidate the capability of hierarchical representation by NetBCE, we visualized the epitopes and non-epitopes using UMAP (Uniform Manifold Approximation and Projection) method based on the feature representation at varied network layers. We found the feature representation came to be more discriminative along the network layer hierarchy. More specifically, the feature representations for epitopes and non-epitopes sites were mixed at the input layer. As the model continues to train, epitopes and non-epitopes tend to occur in very distinct regions with efficient feature representation. 

![image](https://github.com/BioDataStudy/NetBCE/blob/main/Interpretability/github_4.jpg)

# Motif discovery and post-transcriptional regulation
To elucidate the capability of hierarchical representation by NetBCE, we visualized the epitopes and non-epitopes using UMAP (Uniform Manifold Approximation and Projection) method based on the feature representation at varied network layers. We found the feature representation came to be more discriminative along the network layer hierarchy. More specifically, the feature representations for epitopes and non-epitopes sites were mixed at the input layer. As the model continues to train, epitopes and non-epitopes tend to occur in very distinct regions with efficient feature representation. 

![image](https://github.com/BioDataStudy/NetBCE/blob/main/Interpretability/github_4.jpg)

# Usage
Please cd to the NetBCE/prediction/ folder which contains predict.py.
Example: 
```
cd NetBCE/prediction/
python NetBCE_prediction.py -f ../testdata/test.fasta -o ../result/test_result
```
For details of other parameters, run:
```
python NetBCE_prediction.py --help
```
