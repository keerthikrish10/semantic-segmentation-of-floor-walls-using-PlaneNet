# 🚀 PlaneNet-Based Floor and Wall Segmentation


## 📌 Overview
This project is inspired from ofiicial PlaneNet model architecture.Used PlaneNet architecture ,basically a  encoder-decoder CNN-DNN model with RESNET 101  as a base model to build and train the model from scratch.
This project implements **semantic segmentation for floor and wall detection** using **PlaneNet architecture with RESNET 101 as backbone**. It takes indoor images(uploaded or from live cam) as input and predicts segmentation masks, enabling applications in **indoor mapping, AR/VR, and smart home automation,Robot indoor Navigation**.

## ✨ Features
- **Semantic Segmentation:** Predicts masks for floor, wall, and other surfaces.
- ** Model built from scratch:** Uses RESNET 101 as backbone with PlaneNet architecture  for robust segmentation.
- **Performance Metrics:** Computes **Accuracy, IoU, Precision, Recall, and F1 Score**.
- **Flexible Input:** Works **with or without ground truth masks**.
- **Visualization:** Displays segmented output alongside original images.

## 📁 Dataset
- **Dataset Used:** refer DATASET.TXT
- **Structure:**
  ```plaintext
  dataset/
  ├── raw_images/
  │   ├── image_1/
  │   ├── image_2/
  │   ├── ...
  ├── ground_truth/
  │   ├── image1_mask.png
  │   ├── image2_mask.png
  ```

## ⚡ Installation
```bash
# Clone the repo
git clone https://github.com/keerthikrish10/semantic-segmentation-of-floor-walls-using-PlaneNet/
cd PlaneNet-Segmentation

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage
### 1️⃣ **Inference on a Single Image**
```python
python test.py --image_path /path/to/image.png --mask_path /path/to/mask.png  # Optional mask
```

### 2️⃣ **Batch Inference on Multiple Images**
```python
python batch_test.py --input_folder /path/to/images/ --output_folder /path/to/results/
```

### 3️⃣ **Training (If Needed)**
```python
python train.py --epochs 50 --batch_size 16 --dataset_path /path/to/dataset/
```

## 📊 Evaluation Metrics
| Metric    | Description |
|-----------|------------|
| **Accuracy** | Measures the percentage of correctly classified pixels. |
| **IoU** | Intersection over Union for segmentation accuracy. |
| **Precision** | Ratio of correctly predicted positive observations. |
| **Recall** | Model's ability to detect all positive samples. |
| **F1 Score** | Harmonic mean of Precision and Recall. |

## 📷 Example Results
| **Input Image** | **Predicted Segmentation** | 
|---------------|----------------------|--------------------|
| ![Input](https://github.com/keerthikrish10/semantic-segmentation-of-floor-walls-using-PlaneNet/blob/main/streamlitUI-UX.png) | ![Predicted](https://github.com/keerthikrish10/semantic-segmentation-of-floor-walls-using-PlaneNet/blob/main/streamlitprediction.png)|

## 🛠 Technologies Used
- **TensorFlow/Keras** - Model training and inference
- **OpenCV** - Image processing
- **NumPy, Matplotlib** - Data handling and visualization
- **Scikit-learn** - Performance metrics

## 🤝 Approaches used
1. Tried to use official PlaneNet model from https://github.com/art-programmer/PlaneNet . But the code was not updates and has many BUG unfixed
2. Used PlaneNet architecture ,basically a  encoder-decoder CNN-DNN model with RESNET 101  as a base model to build and train the model from scratch.

##Further optimizations needed
1.Use PYtorch GPU accelerated with CUDA 
2.transfer learning with complex models mask RCNN OR Monocular Depth Estimation Models (MiDaS, DPT)
3. Deploying in AWS/GCP

## obtained performance scores 
training accuracy : 90% ; validation accuracy : 89% ; loss = 0.2
testing:
🔹 Accuracy: 0.8807
🔹 Precision: 0.6799
🔹 Recall: 0.9558
🔹 F1-score: 0.7946
🔹 IoU: 0.6592
## 📜 License
This project is **MIT licensed**. Feel free to use and improve it!

## 📬 Contact
For queries, reach out to **[keerthi krishna ](https://github.com/keerthikrish10)** ✉️

