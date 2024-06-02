# Live Person Identification
A machine learning project that identifies and names a person in CCTV footage from various angles.

## Action plan
### 1. Define the Problem
Objective: To build a model that can identify and name a person in CCTV footage from various angles.
Input: CCTV images or video frames.
Output: The name of the identified person.

### 2. Data Collection
Dataset: Collect a dataset of images or video frames of the people you want to recognize. Ensure the dataset includes images from various angles.
Annotations: Label the images with the names of the people.
#### Data Collection - the process 
Sources: Use publicly available datasets like LFW (Labelled Faces in the Wild), or collect your own data.  
Storage: Organize the data into directories named after the identities  

### 3. Data Preprocessing
Face Detection: Use a face detection algorithm (like MTCNN or OpenCV's Haar Cascades) to detect faces in the images.
Face Alignment: Align the detected faces to a standard orientation.
Data Augmentation: Apply augmentation techniques like rotation, scaling, and flipping to increase the dataset size and variability.

### 4. Model Selection
Pre-trained Models: Utilize pre-trained models like VGG-Face, FaceNet, or Dlib for face recognition.
Fine-tuning: Fine-tune the pre-trained models on your dataset.

### 5. Training the Model
Feature Extraction: Use the pre-trained model to extract features from the faces.
Classifier: Train a classifier (e.g., SVM, KNN, or a neural network) on the extracted features to classify the identities.

### 6. Model Evaluation
Metrics: Evaluate the model using metrics such as accuracy, precision, recall, and F1-score.
Cross-validation: Use cross-validation to ensure the model’s robustness.

### 7. Real-time Inference: 
Implement the model for real-time inference on CCTV footage.
Optimization: Optimize the model for speed and accuracy, potentially using techniques like model quantization.
