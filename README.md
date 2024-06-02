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

### 7. Deployment
Real-time Inference: Implement the model for real-time inference on CCTV footage.  
Optimization: Optimize the model for speed and accuracy, potentially using techniques like model quantization.  
Real-time Detection: Capture frames from CCTV and pass through the detection, alignment, and recognition pipeline.  
Optimization: Use libraries like TensorRT for optimizing inference.  

Tools and Libraries  
TensorFlow/Keras: For building and training models.  
OpenCV: For image processing.  
MTCNN: For face detection.  
Dlib: For face recognition models and utilities.  
Scikit-learn: For training classifiers.  
For a face recognition project, you can utilize several publicly available datasets. Here are some popular datasets you can use:

### Public Face Datasets

1. **Labeled Faces in the Wild (LFW)**
   - **Description**: Contains more than 13,000 labeled images of faces from the web.
   - **Use**: Primarily used for studying the problem of unconstrained face recognition.
   - **Link**: [LFW Dataset](http://vis-www.cs.umass.edu/lfw/)

2. **VGGFace2**
   - **Description**: A large-scale dataset with over 3.3 million images of more than 9,000 individuals.
   - **Use**: Suitable for training deep learning models for face recognition.
   - **Link**: [VGGFace2 Dataset](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/)

3. **CelebA**
   - **Description**: Contains more than 200,000 celebrity images with 40 attribute labels per image.
   - **Use**: Useful for attribute prediction and face recognition.
   - **Link**: [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

4. **MS-Celeb-1M**
   - **Description**: A large dataset with 10 million images of 100,000 celebrities.
   - **Use**: Ideal for large-scale face recognition tasks.
   - **Link**: [MS-Celeb-1M Dataset](https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-million-celebrities/)

5. **FaceScrub**
   - **Description**: Contains over 100,000 images of 530 individuals.
   - **Use**: Suitable for face recognition and verification tasks.
   - **Link**: [FaceScrub Dataset](http://vintage.winklerbros.net/facescrub.html)

6. **IMDB-WIKI**
   - **Description**: The largest public dataset of face images with age and gender labels, containing over 500,000 images.
   - **Use**: Useful for training age and gender prediction models, as well as face recognition.
   - **Link**: [IMDB-WIKI Dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)

### Custom Dataset Collection

If the available datasets do not meet your specific requirements, you might need to collect your own dataset:

1. **Capture Images**: Use a camera to capture images of the individuals you want to recognize, ensuring various angles, lighting conditions, and expressions.
2. **Labeling**: Manually label the images with the corresponding identities.
3. **Data Augmentation**: Apply data augmentation techniques to increase the variability in the dataset.

### Data Preprocessing

Regardless of the dataset you choose, you'll need to preprocess the images to prepare them for training. This includes face detection, alignment, and normalization. Here's a basic example using MTCNN and OpenCV for preprocessing:

```python
import cv2
import numpy as np
from mtcnn import MTCNN

# Initialize MTCNN face detector
detector = MTCNN()

def preprocess_image(image_path):
    # Load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    results = detector.detect_faces(image_rgb)
    if results:
        # Extract bounding box from the first face
        x, y, width, height = results[0]['box']
        face = image_rgb[y:y+height, x:x+width]
        
        # Resize to a fixed size (e.g., 160x160 pixels)
        face = cv2.resize(face, (160, 160))
        return face
    return None

# Example usage
image_path = 'path_to_image.jpg'
preprocessed_face = preprocess_image(image_path)
if preprocessed_face is not None:
    cv2.imshow("Preprocessed Face", preprocessed_face)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

### Summary

- **Choose a dataset**: Select from publicly available datasets like LFW, VGGFace2, CelebA, MS-Celeb-1M, FaceScrub, or IMDB-WIKI.
- **Collect a custom dataset**: If needed, capture images and label them manually.
- **Preprocess the data**: Use face detection and alignment techniques to prepare the images for model training.

By following these steps, you can acquire a suitable dataset for your face recognition project.
Both MTCNN (Multi-task Cascaded Convolutional Networks) and OpenCV's Haar Cascades are popular face detection methods, but they have different strengths and use cases. Here’s a comparison to help you decide which one to use for your face recognition project:

### MTCNN

#### Advantages:
1. **Accuracy**: MTCNN is generally more accurate and robust compared to Haar Cascades. It performs well with various lighting conditions, occlusions, and facial expressions.
2. **Detection of Facial Landmarks**: MTCNN not only detects faces but also identifies facial landmarks (eyes, nose, mouth), which is useful for face alignment.
3. **Deep Learning-Based**: Utilizes deep learning, which tends to perform better with more complex and variable data.

#### Disadvantages:
1. **Performance**: Slower than Haar Cascades due to its complexity and deep learning architecture.
2. **Dependencies**: Requires more dependencies and setup (e.g., TensorFlow or PyTorch).

### OpenCV's Haar Cascades

#### Advantages:
1. **Speed**: Faster and less computationally intensive, making it suitable for real-time applications on devices with limited resources.
2. **Ease of Use**: Simple to use with OpenCV, which is widely adopted and well-documented.
3. **Lightweight**: Requires fewer resources compared to deep learning-based methods.

#### Disadvantages:
1. **Accuracy**: Generally less accurate and more prone to false positives and negatives, especially in challenging conditions (e.g., poor lighting, occlusions).
2. **Limited Functionality**: Primarily detects faces without providing detailed landmarks for alignment.

### Use Cases

- **Use MTCNN if**:
  - You need high accuracy and robustness.
  - You require facial landmarks for alignment.
  - Your application can tolerate slightly slower performance and has access to sufficient computational resources.

- **Use OpenCV's Haar Cascades if**:
  - You need fast performance and low computational overhead.
  - You are working in controlled environments with good lighting and minimal occlusions.
  - You want a simple and easy-to-implement solution.

### Implementation Examples

#### MTCNN

```python
from mtcnn import MTCNN
import cv2

# Initialize MTCNN detector
detector = MTCNN()

def detect_faces_mtcnn(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(image_rgb)
    for result in results:
        x, y, width, height = result['box']
        cv2.rectangle(image, (x, y), (x+width, y+height), (255, 0, 0), 2)
    cv2.imshow("MTCNN Face Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = 'path_to_image.jpg'
detect_faces_mtcnn(image_path)
```

#### OpenCV's Haar Cascades

```python
import cv2

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces_haar(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.imshow("Haar Cascade Face Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = 'path_to_image.jpg'
detect_faces_haar(image_path)
```

### Conclusion

For your project involving identifying and naming people in CCTV footage, **MTCNN** would generally be the better choice due to its higher accuracy and robustness, especially when dealing with various angles and challenging conditions. However, if performance is a critical factor and the environment is controlled, **OpenCV's Haar Cascades** can be a viable option.
