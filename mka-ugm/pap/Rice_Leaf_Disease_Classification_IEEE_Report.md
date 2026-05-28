# Rice Leaf Disease Classification

**Ahmad Nuhisya Adillaumam** (25/563396/PPA/07111), **Muhammad Zidan Arsyad** (25/572381/PPA/07192)

## Abstract—
This report presents an improved rice leaf disease classification pipeline that combines handcrafted texture and color features with fine-tuned deep features using a weighted fusion strategy. The proposed method extracts disease region crops from YOLO annotations, computes a diverse handcrafted feature set (HOG, multi-scale LBP, Haralick, color histogram, color moments, and Gabor filters), and obtains domain-adapted deep embeddings from a fine-tuned ResNet backbone. Mutual information is used to select the most informative handcrafted features, and a late fusion stacking strategy is applied to improve classification robustness. The experimental design includes SVM, MLP, Random Forest, and stacking meta-learners, enabling comprehensive comparison across multiple architectures.

## Index Terms—
Rice leaf disease classification, handcrafted features, deep learning, feature fusion, mutual information, ResNet, stacking ensemble.

## I. Introduction
Rice leaf disease detection is a critical problem in precision agriculture. Early diagnosis allows targeted treatment, minimizing crop loss and increasing yield. Traditional image-based systems rely on handcrafted features extracted from leaf textures and color patterns, while modern approaches use deep convolutional neural networks to learn discriminative representations automatically.

This work proposes a hybrid classification pipeline that fuses handcrafted and deep features to exploit complementary strengths. The pipeline builds upon an existing rice leaf disease dataset in YOLO annotation format and extends the baseline by adding fine-tuning, mutual information feature selection, weighted fusion, and stacking meta-learners.

## II. Related Work
Several studies have used texture descriptors and deep convolutional features for plant disease detection. Handcrafted descriptors such as local binary patterns (LBP), histograms of oriented gradients (HOG), Haralick features, and Gabor filters are effective for capturing leaf surface irregularities. Deep learning models such as ResNet and EfficientNet can learn higher-level disease characteristics, but they often benefit from domain-specific fine-tuning.

Hybrid fusion approaches have shown improved performance in other agricultural image classification tasks by combining handcrafted and deep features. Mutual information-based selection is a common technique to reduce dimensionality while preserving discriminative information.

## III. Proposed Method
### A. Dataset and Preprocessing
The dataset is the RICE Leaf Diseases dataset from Kaggle, provided in YOLO format. The pipeline extracts bounding box crops from the training and validation splits using YOLO annotations. For each crop, the pipeline ensures a minimum size and balances the number of samples per class by limiting the maximum number of crops per class.

### B. Handcrafted Feature Extraction
Handcrafted features are extracted from cropped leaf regions resized to 128×128 pixels. The feature set includes:
- Color histograms in HSV space.
- HOG descriptors capturing edge orientation patterns.
- Multi-scale LBP for texture analysis across three scales.
- Haralick features from gray-level co-occurrence matrices.
- Color moments (mean, standard deviation, skewness).
- Gabor filter responses at multiple orientations and scales.

This rich feature set produces a high-dimensional handcrafted representation designed to capture both color and texture disorders associated with various rice diseases.

### C. Deep Feature Extraction and Fine-Tuning
The deep branch uses a pretrained backbone model, selected as ResNet50 in the notebook. The last residual block and classification head are fine-tuned on rice leaf disease crops with data augmentation, including random resized crop, flips, rotation, color jitter, and random grayscale.

After fine-tuning, deep features are extracted from the penultimate layer of the network. The deep feature vectors are normalized with StandardScaler before fusion.

### D. Feature Selection and Fusion
Mutual information is computed between handcrafted features and class labels. The top 70% of handcrafted features are selected, significantly reducing the dimensionality while preserving class-relevant information.

Deep features are scaled by a configurable weight factor to compensate for the dimensionality imbalance between handcrafted and deep representations. The fused feature vector concatenates MI-selected handcrafted features and weighted deep features.

PCA is optionally applied to the fused vector to retain 95% of variance, further reducing dimensionality for classifier training.

### E. Classification and Stacking
The pipeline trains several classifiers on different feature combinations:
- SVM on MI-selected handcrafted features.
- SVM on fine-tuned deep features.
- SVM on weighted fused features.
- SVM with grid search on fused features.
- MLP on fused features.
- Late fusion stacking with SVM base learners and Logistic Regression meta-learner.
- Enhanced late fusion with additional Random Forest base learner and MLP meta-learner.
- Random Forest on fused features.

The stacking strategy uses probability outputs from base classifiers as meta-features, enabling a robust ensemble decision.

## IV. Experimental Setup
### A. Implementation Details
The pipeline is implemented in Python using OpenCV, scikit-image, scikit-learn, PyTorch, and torchvision. The notebook runs on a machine with GPU support, and training uses AdamW optimization with cosine annealing learning rate scheduling.

### B. Parameters
Key configuration settings include:
- Backbone: ResNet50.
- Fine-tuning epochs: 15.
- Learning rate: 1e-4.
- Batch size: 32.
- MI selection percentile: 70%.
- Deep fusion weight: 3.0.
- PCA variance retention: 95%.

### C. Evaluation Metrics
The evaluation metrics include accuracy, precision, recall, F1-score, and confusion matrices. The notebook also generates per-class performance charts, feature importance visualizations, and t-SNE plots of the fused feature space.

## V. Results
The pipeline compares eight classification experiments and identifies the best-performing model based on validation accuracy. The baseline pipeline reported in the notebook has a best accuracy of 80.21% using a frozen ResNet18 model.

The improved pipeline is designed to surpass this baseline by employing:
- fine-tuning of the backbone,
- more expressive handcrafted descriptors,
- mutual information feature selection,
- weighted fusion of complementary feature groups,
- stacking ensemble learning.

### A. Model Comparison
The notebook prepares a comparison bar chart of all models and highlights the best performing pipeline. Confusion matrix visualizations are used to inspect class-wise errors, and per-class precision/recall/F1 bars are plotted for deeper analysis.

### B. Feature Contribution Analysis
Mutual information scores are aggregated by handcrafted feature group. The analysis shows which groups contribute most to discrimination and how many features from each group are retained after MI selection.

### C. Ablation Study
An ablation study evaluates the impact of each major improvement:
- removing MI selection,
- removing deep feature weighting,
- using frozen ResNet18 without fine-tuning,
- disabling enhanced handcrafted features (Gabor and multi-scale LBP).

This study quantifies the contribution of each component to the overall accuracy.

## VI. Conclusion
This report describes an improved rice leaf disease classification pipeline that integrates handcrafted texture/color features with fine-tuned deep features. The method emphasizes feature selection and weighted fusion to balance heterogeneous representations, while the ensemble stacking strategy further improves classification robustness.

Future work may include:
- using larger fine-tuned backbones such as ResNet101 or EfficientNet,
- applying more advanced occlusion-aware augmentation,
- exploring end-to-end trainable fusion networks,
- conducting cross-dataset validation to measure generalization.

## References
[1] A. Krizhevsky, I. Sutskever, and G. E. Hinton, “ImageNet classification with deep convolutional neural networks,” *Advances in Neural Information Processing Systems*, 2012.

[2] K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” *International Conference on Learning Representations*, 2015.

[3] R. O. Duda, P. E. Hart, and D. G. Stork, *Pattern Classification*, 2nd ed., Wiley, 2000.

[4] N. Dalal and B. Triggs, “Histograms of oriented gradients for human detection,” *IEEE Computer Society Conference on Computer Vision and Pattern Recognition*, 2005.

[5] T. Ojala, M. Pietikainen, and D. Harwood, “Performance evaluation of texture measures with classification based on Kullback discrimination of distributions,” *Pattern Recognition*, 1996.
