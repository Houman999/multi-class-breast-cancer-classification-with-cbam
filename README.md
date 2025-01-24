# multi-class breast cancer classification with cbam
 **What is Breast Cancer and Type of that:**

Breast cancer is a complex and progressive disease that develops in the breast tissues, often originating from the milk-producing lobules or the ducts that carry milk to the nipple. The image below provides a comprehensive overview of the stages of breast cancer development and highlights the transition from benign to malignant forms. This process begins with normal cells lining the duct walls, which may gradually develop abnormalities (atypical hyperplasia) and eventually develop into ductal carcinoma in situ (DCIS), where the cancer cells are confined within the duct. As the disease progresses, cancer cells break through the duct walls, become invasive, and in severe cases, metastatic cancer develops and spreads to other parts of the body through the blood and lymphatic system.

![Progression of the Breast Cancer](Images/Screenshot%202025-01-24%20155319.png)


**Benign breast tumors**
Benign breast tumors are non-cancerous tumors that can develop in breast tissue. The following are distinct histological types of benign breast tumors:

**Adenose**
Adenosis is characterized by an increase in the number of glandular structures in the breast tissue. It can sometimes mimic cancer but is generally considered benign.
**fibroadenoma**
Fibroadenomas are the most common benign breast tumors, especially in young women. They are composed of both glandular and stromal (connective) tissue, which are well defined as masses on imaging studies.
**Phylloid tumor**
Phyllode tumors, also known as phyllodes cystosarcomas, are rare fibroepithelial tumors that can be benign or malignant. They usually grow quickly and can be large at the time of diagnosis. Histologically, they show a leaf-like architecture.
**Tubular adenoma**
Tubular adenomas are rare benign tumors composed primarily of tubular structures. They may be confused with fibroadenomas, but have distinct histological features, including elongated ducts and minimal stromal component.

**Malignant breast tumors**
Malignant breast tumors are cancerous tumors that can invade surrounding tissues and metastasize to other parts of the body. The following are distinct histological types of malignant breast tumors:

**ductal carcinoma**
Ductal carcinoma is the most common type of breast cancer that originates from the milk ducts. It can be further classified into invasive and non-invasive forms (ductal carcinoma in situ). Invasive ductal carcinoma (IDC) tends to spread beyond the ducts into surrounding breast tissue.
**Lobular carcinoma**
Lobular carcinoma arises from the lobules (milk-producing glands) of the breast. Similar to ductal carcinoma, it can be invasive or non-invasive. Invasive lobular carcinoma often appears as a subtle thickening rather than a distinct mass.
**Mucinous carcinoma**
Mucinous carcinoma is a rare subtype characterized by the presence of mucin-producing cancer cells. This cancer has a better prognosis than other types of invasive breast cancer due to its slower growth rate.
**Papillary carcinoma**
Papillary carcinoma is another rare form of breast cancer that usually appears as a well-defined mass with a papillary structure under microscopic examination. It can be invasive or non-invasive and is often associated with better results compared to more invasive types.

## Project Overview

The BreakHis - Breast Cancer Histopathological Dataset is a critical resource for medical image analysis, designed to advance research in breast cancer classification. This dataset consists of high-resolution histopathological images of breast tissue, specifically curated for multi-class classification tasks. Researchers and developers can utilize this dataset to train and evaluate machine learning models, contributing to advancements in cancer diagnostics.

**Dataset Structure**
- The dataset is organized into:

- - Classification Type: Multi-class classification, focusing on 8 distinct tumor types.
- - Magnification Levels: Images are available at varying levels of zoom: 40X, 100X, 200X, and 400X.

/dataset/
    ├── magnification_level/
    │      ├── 40X/
    │      ├── 100X/
    │      ├── 200X/
    │      ├── 400X/
    ├── class_name/
           ├── adenosis/
           ├── ductal_carcinoma/
           ├── fibroadenoma/
           ├── lobular_carcinoma/
           ├── mucinous_carcinoma/
           ├── papillary_carcinoma/
           ├── phyllodes_tumor/
           ├── tubular_adenoma/

**Imbalance Handling**
- Class imbalance in the dataset is addressed using a combination of oversampling and undersampling techniques 

![Plot Imbalance dataset](Images/Screenshot%202025-01-24%20170208.png)

**Image Preprocessing**
-  Images are preprocessed to enhance quality and standardize input for deep learning models:

- - Contrast Enhancement: The CLAHE (Contrast Limited Adaptive Histogram Equalization) technique is applied to improve image contrast, particularly useful for histopathological images.
- - Normalization: Images are normalized to the [0, 1] range for better compatibility with deep learning models.
- - Resizing: All images are resized to a fixed target size (e.g., 224x224 pixels) for consistency.
- - Data Augmentation: Random horizontal and vertical flips are applied during training to increase model generalization and robustness.
- - These steps ensure the dataset is well-prepared for training robust and high-performance classification models.

![Histogram](Images/Screenshot%202025-01-24%20155757.png)

![Sample of Images](Images/Screenshot%202025-01-24%20155819.png)


**Model Architecture with CBAM**
The model is built on top of InceptionResNetV2, leveraging pre-trained weights for feature extraction while incorporating the CBAM layer for attention-based refinement.

- CBAM Layer
- - The Convolutional Block Attention Module (CBAM) is integrated into the model to enhance feature refinement by focusing on critical spatial and channel-wise information.

- - Channel Attention Module:
- - - Computes attention weights along the channel dimension using global average pooling and global max pooling.
- - - These pooled features are processed by shared dense layers and combined to generate channel-wise attention weights.
- - Spatial Attention Module:
- - - Focuses on spatial relationships by combining average and max pooling along the channel axis.
- - - A convolutional layer applies attention across the spatial dimensions.
- - Feature Refinement:
- - - The input features are refined by element-wise multiplication with the channel and spatial attention maps, enabling the model to focus on important regions and suppress irrelevant features.
- - - This dual attention mechanism improves the model's ability to capture complex patterns in histopathological images, leading to better classification performance.

## Results 
The proposed model, incorporating the CBAM and the InceptionResNetV2 backbone, achieved a 94% accuracy on the validation dataset. The model demonstrates a strong ability to learn and generalize from the dataset, as reflected in the training and validation metrics.

Training Performance
The performance of the model during training and validation is visualized in the figure below, which includes:

Accuracy Plot: Shows the progression of training and validation accuracy across 50 epochs. The training accuracy steadily increases and converges near 1.0, while the validation accuracy reaches approximately 0.94, indicating good generalization without severe overfitting.
Loss Plot: Depicts the reduction in training loss over time and the validation loss fluctuation. The validation loss initially decreases but stabilizes around a consistent value after early epochs.
The close alignment of training and validation curves demonstrates that the implemented techniques, including dataset balancing, augmentation, and CBAM, contributed effectively to minimizing overfitting and improving the model's robustness.

![Plot Training and Validation Metrics ](Images/Screenshot%202025-01-24%20160046.png)

## Conclusion
Conclusion
This project demonstrates the potential of using advanced deep learning techniques for multi-class breast cancer classification using histopathological images. By integrating the CBAM with the InceptionResNetV2 backbone, the model effectively enhances feature extraction and attention to critical regions, leading to improved classification performance.

The inclusion of robust preprocessing steps, such as contrast enhancement, normalization, and data augmentation, ensures high-quality inputs to the model while addressing the inherent class imbalance in the dataset through oversampling and undersampling methods. These efforts contributed to a final validation accuracy of 94%, highlighting the model’s ability to generalize well to unseen data.

This work emphasizes the importance of attention mechanisms in medical imaging and lays a strong foundation for future research. By further exploring techniques such as fine-tuning specific domain data, experimenting with different architectures, and using transfer learning, the performance of such models can be further improved. Ultimately, the methods presented in this project are a valuable tool for advancing breast cancer diagnosis and facilitating early detection, which is crucial for improving patient outcomes.
Also, visit my Kaggle account for access to all the code, including data visualizations, confusion matrices, and ROC plots, as well as model output.

![My Kaggle](https://www.kaggle.com/code/manofnoego/multi-class-breast-cancer-classification-with-cbam)

## Reference

- Lane, Deanna L, Malai Muttarak and Wei Tse Yang. “Malignant Breast Tumors.” (2013).
- Zhao, Shuang, Guohui Wei, Zhiqing Ma and Wenhua Zhao. “Breast Tumors Multi-classification Study Based on Histopathological Images with Radiomics Approach.” IOP Conference Series: Earth and Environmental Science 440 (2020): n. pag.
- Agrawal, Priyanka, Suresh Kumar Sutrakar, Jagannath Jatav, Parul Singh Rajpoot, Shambhavi, Sadhana Yadav and Pushpkunjika Sharma. “Histopathological Spectrum of Benign and Borderline Breast Lesions: A Crosssectional Study from Vindhya Region, Madhya Pradesh, India.” JOURNAL OF CLINICAL AND DIAGNOSTIC RESEARCH (2024): n. pag.
- Abunasser, Basem S., Mohammed Rasheed J. Al-Hiealy, Ihab S. Zaqout, and Samy S. Abu-Naser. "Convolution neural network for breast cancer detection and classification using deep learning." Asian Pacific journal of cancer prevention: APJCP 24, no. 2 (2023): 531.
- Khan, Saif ur Rehman, Asif Raza, Inzamam Shahzad and Hafiz Muhammad Ijaz. “Deep transfer CNNs models performance evaluation using unbalanced histopathological breast cancer dataset.” Lahore Garrison University Research Journal of Computer Science and Information Technology (2024): n. pag.
- https://medium.com/visionwizard/understanding-attention-modules-cbam-and-bam-a-quick-read-ca8678d1c671
