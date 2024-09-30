# 🎓 Data Scientist Portfolio

---

## 📊 Projects

### Project 1: Credit Card Fraud Detection

**Objective** <br/> 
To determine which model performs best when data is reduced or augmented.

**Technologies Used** <br/>
- Dimensionality Reduction: PCA, tSNE, UMAP
- Dimensionality Augmentation: SMOTE, BorderLineSMOTE, ADASYN
- Machine Learning Models: RandomForest, XGBoost, CatBoost, LightGBM
- Deep Learning Models: TensorFlow, Pytorch 

**Key Results** <br/>
To compare whether dimensionality reduction or augmentation improves model performance, <br/>
I used various machine learning and deep learning models. <br/>
As a result, I was able to create a ranking table showing which method and model combination yielded the best performance. <br/>
The accuracy was similar, so I ranked them based on the ROC_AUC_SCORE.

**URL** <br/>
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

### Project 2: YOLOv10 Pretrained vs Custom

**Objective** <br/>
To compare which performs better between the pretrained and custom YOLOv10 models.

**Technologies Used** <br/>
- Model: YOLOv10
- Package: ultralytics, supervision, cv2

**Key Results** <br/>
After capturing the video and creating multiple frames, <br/>
each frame was trained with the model, and then these frames were reassembled into a single video. <br/>
For the pretrained model, predictions were made directly using the model. <br/>
For the custom model, pre-prepared data was trained using the original YOLOv10 weights, <br/>
and the best weights obtained were selected as the final weights for the model, which was then used for predictions. <br/>
This process is similar to a relay race.

When comparing the pretrained and custom models, there was a significant difference. <br/>
The custom model, which was provided with images of various classes consistently, had a broader prediction range than the automatically recognizing pretrained model. <br/>
However, its accuracy was much lower compared to the pretrained model.

**URL** <br/>
https://github.com/THU-MIG/yolov10 <br/>
https://docs.ultralytics.com/ko/models/yolov10

---

### Project 3: Detectron2 Pretrained vs Custom

**Objective** <br/>
To compare which performs better between the pretrained and custom Detectron2 models.

**Technologies Used** <br/>
- Model: Detectron2
- Package: detectron2, cv2

**Key Results** <br/>
Detectron2 is almost identical to YOLOv10, but there are two key differences. <br/>
First, Detectron2 uses Faster RCNN weights, unlike YOLOv10. <br/>
Second, while YOLOv10 shows some differences in results between pretrained and custom models, <br/>
Detectron2 exhibits no noticeable differences."

**URL** <br/>
https://github.com/facebookresearch/detectron2/blob/main/README.md

---

### Project 4: AI Cover - RVC

**Objective** <br/>
Using the RVC model to make one singer's voice sing another singer's song.

**Technologies Used** <br/>
- Model: RVC

**Key Results** <br/>
This project can be explained in five steps. <br/>
First, split the downloaded YouTube music into vocals and background music. <br/>
Second, slice the vocals into multiple segments to enhance the model's learning. <br/>
Third, download the RVC_pretrained model. <br/>
Fourth, train the model. <br/>
Fifth, generate a music file where the singer performs a different song. <br/>

I was amazed at how natural the generated music sounded. <br/>
Detailed adjustments can be made, and having an expert involved could further improve the synchronization and overall quality. 

**URL** <br/>
https://github.com/facebookresearch/demucs <br/>
https://github.com/openvpi/audio-slicer

---

### Project 5: CNN - CIFAR-10

**Objective** <br/>
Using CIFAR-10 data, build a complex CNN with TensorFlow and PyTorch.

**Technologies Used** <br/>
- Models : TensorFlow, Pytorch
- CNN Process : Data Augmentation, Conv2d, Padding, Batch Normalization, Pooling, Dropout, Flatten 

**Key Results** <br/>
All processes of the CNN with TensorFlow and PyTorch are included: Data Augmentation, Padding, Batch Normalization, Pooling, Dropout, Flatten.

**URL** <br/>
https://www.cs.toronto.edu/~kriz/cifar.html

---

### Project 6: CLIP

**Objective** <br/>
To Find out how to use CLIP Model(Zero-shot image classification model) on Web Images and images from computer storage.

**Technologies Used** <br/>
- Model : CLIP
- Skill: Zero-shot image classification

Zero-shot image classification is a technique where the model can correctly classify new images even if it hasn't been directly trained on images of a specific class during training. The model leverages pre-learned knowledge and similarities or relationships between different classes it has learned to infer new classes.

CLIP (Contrastive Language-Image Pretraining) is a representative model for zero-shot image classification. It simultaneously learns from both images and text, allowing it to understand the relationship between the two. CLIP employs contrastive learning, where it pairs images with their corresponding text descriptions during training. This enables the model to associate previously unseen classes with appropriate text descriptions, allowing for effective zero-shot classification.
  
**Key Results** <br/>
The results of web images and images from computer storage predicted by CLIP.

**URL** <br/>
https://github.com/openai/CLIP

---

### Project 7: SAM2

**Objective** <br/>
After detecting objects using YOLO, SAM2 is used to generate segmentation masks for the detected objects, and then the masks are overlaid with colors corresponding to each object to create an output video. YOLO generates bounding boxes, and SAM processes them to handle segmentation, integrating the two models.

**Technologies Used** <br/>
Models: SAM2, YOLO

**Key Results** <br/>
A new video detected by the model.

URL <br/>
https://github.com/facebookresearch/segment-anything-2 <br/>
https://docs.ultralytics.com/ko/models/yolov10

---

## 📈 Skills

- **Programming Languages**: Python
- **Data Preprocessing**: Pandas, NumPy
- **Data Visualization**: Matplotlib
- **Machine Learning & Deep Learning**: Scikit-Learn, TensorFlow, Pytorch, OpenCV
- **Databases**: 
- **Tools**: Jupyter Notebook, Google Colab

---

## 🛠️ Tools & Technologies

<p>
  <img src="https://img.shields.io/badge/Tensorflow-FF6F00.svg?style=for-the-badge&logo=Tensorflow&logoColor=white" alt="Tensorflow" width="120" height="30"/>
  <img src="https://img.shields.io/badge/Pytorch-EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white" alt="Pytorch" width="120" height="30"/>
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8.svg?style=for-the-badge&logo=OpenCV&logoColor=white" alt="OpenCV" width="120" height="30"/>
</p>

---

