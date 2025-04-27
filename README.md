# Plant Disease Detection using Custom CNN

### Name-Vipul Chopra 
**RollNo.-20221284**

This project implements a custom Convolutional Neural Network (CNN) for **multi-class classification** of plant leaf diseases using the **New Plant Diseases Dataset (Augmented)**.  
The model achieves good classification accuracy without using any transfer learning models like ResNet or VGG.

## 📁 Project Structure
        project_Vipul_Chopra/
    ├── checkpoints/
    │   └── final_weights.pth        # Trained model weights
    ├── data/
    │   └── Class_1/                  # Subfolder with Class name 
            └──img01.jpg, img02.jpg.. # Sample images for testing
    ├── dataset.py                    # Custom dataset and dataloader
    ├── model.py                      # CNN model architecture
    ├── train.py                      # Training and evaluation scripts
    ├── predict.py                    # Prediction script (batch inference)
    ├── config.py                     # Configuration (hyperparameters, paths)
    ├── test_dataset.py               # Script to test dataloading
    ├── environment.yml               # Conda environment file
    └── README.md                     # This file

## 📌 Important Instructions

- The `environment.yml` file is already provided inside the project directory.  
- It contains all the necessary packages needed to successfully run this project.
- When you create the environment using:
  ```bash
  conda env create -f environment.yml
  conda activate DS_course
- I have used CUDA for training the model faster, so the pytorch package is a CUDA-compatible one. 


---

###  Full List of Packages from your environment.yml :

| Category        | Packages                                    |
|-----------------|---------------------------------------------|
| Deep Learning   | `torch`, `torchvision`, `torchaudio`        |
| Core Libraries  | `numpy`, `matplotlib`, `pillow`, `tqdm`     |
| Testing         | `pytest`, `pytest-shutil`                   |
| Others          | `python`, `pip`, `setuptools`, `wheel`      |




##  Testing Details

- The `data/` folder contains **38 subfolders**, each corresponding to `Class_1/` a different plant disease class.
- Each subfolder contains **10 images** related to that specific class.
- The image filenames themselves **do not contain class labels**.
- The class of an image is determined by **the name of the subfolder** it is stored in.
- During prediction, the model infers the label, and we compare it based on the folder name where the image originally belonged.


---
## Some Important Points
- [The link for the dataset used in the project](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset/data)
- In the original dataset provided at the Kaggle link, the `test/` folder contains images where the **class information is embedded in the filename** itself, e.g., `Apple___Apple_scab_1.jpg`.
- Therefore, if testing on that Kaggle test set, you would need to **extract the label from the filename** while evaluating the predictions.
- On that **test** folder of official dataset, model outputs-
    ```python
    Found 33 test images.

    Doing testing on 5 random samples for example purpose-

    PotatoHealthy1.JPG ➜ Predicted: Potato___healthy | Expected: Potato___healthy
    AppleCedarRust1.JPG ➜ Predicted: Apple___Cedar_apple_rust | Expected: Apple___Cedar_apple_rust
    AppleScab2.JPG ➜ Predicted: Apple___Apple_scab | Expected: Apple___Apple_scab
    TomatoYellowCurlVirus1.JPG ➜ Predicted: Tomato___Tomato_Yellow_Leaf_Curl_Virus | Expected: Tomato___Tomato_Yellow_Leaf_Curl_Virus
    TomatoEarlyBlight6.JPG ➜ Predicted: Tomato___Early_blight | Expected: Tomato___Early_blight

    Accuracy: 100.00% (5/5)
    ```
- If checked on the whole test model-
    ```python
    Found 33 test images.

    Doing testing on whole test folder-

    Accuracy: 90.91% (30/33)
    ```
- However, in this project, the `data/` folder is organized differently:
  - It contains **subfolders** for each class (total 38 classes).
  - Each subfolder contains 10 images corresponding to that class.
  - In this structure, the **true label is determined by the folder name**, not by parsing the image filename.
- In the config file the path for dataset is given that of **data** folder of the project
    ```python
    MODEL_WEIGHTS_PATH = r"checkpoints\final_weights.pth"
    SAMPLE_IMAGE_DIR ="data"
    ```
- In the given Kaggle dataset, the `train/` and `valid/` folders are already provided separately.
- If you are using only the train/ folder from the Kaggle dataset, everything will work perfectly fine (it is large enough).
- But if you want to use both the train/ and valid/ folders together, you need to combine them into one dataset first (keeping the same subfolder class structure), before splitting inside the code.
- However, in this project setup, we load the dataset and then manually perform an **80-20 train-validation split** using the `random_split()` method
    ```python
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    ```
- If training is started again the model weights will now be saved in **TRAIN_WEIGHTS_PATH** which is defined in **config.py** and to run model on these new weigths you have to replace this path into **MODEL_WEIGHTS_PATH**.This is done to ensure nobody accidentlly overwrites the original `final_weights.pth` file

## Expected Output and Accuracy

- The `predict.py` file contains a `classify_plants()` function that accepts a **batch** (list) of image file paths as input.
- The function returns a **list of predicted class labels** corresponding to each input image.
- It will **automatically load trained model weights** from `checkpoints/final_weights.pth` if available.
- If the weights are not found, it will use the **untrained (raw) model** and issue a warning.
- Expected label list returned by **predict.py** on sample batch-
    ```python
    ['Corn_(maize)___healthy', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Potato___healthy', 'Strawberry___Leaf_scorch', 'Potato___Late_blight', 'Strawberry___Leaf_scorch', 'Apple___Cedar_apple_rust', 'Tomato___Tomato_mosaic_virus', 'Apple___Black_rot', 'Tomato___Target_Spot', 'Apple___healthy', 'Strawberry___Leaf_scorch', 'Grape___healthy', 'Cherry_(including_sour)___healthy', 'Pepper,_bell___healthy', 'Pepper,_bell___Bacterial_spot', 'Orange___Haunglongbing_(Citrus_greening)', 'Grape___healthy', 'Corn_(maize)___healthy', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___healthy', 'Tomato___Leaf_Mold', 'Pepper,_bell___healthy', 'Peach___healthy', 'Apple___healthy']
    ```
- Accuracy test results on the **data** folder provided gives expected return like-
    ```python
    Found 35144 images across 38 classes.
    ✅ Loaded model weights from checkpoints\final_weights.pth
    Found 35144 images across 38 classes.
    Accuracy on 380 sampled images: 93.95%
    ```

