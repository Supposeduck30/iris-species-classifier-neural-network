# Iris Species Classifier Neural Network
## A fully functional neural network written in Python using that learns how to identify different iris species
This project demonstrates a neural network built with TensorFlow that learns how to classify iris flowers into one of three species based on sepal and petal measurements. This project:
- Implements an iris dataset with 150 samples
- Trains a neural network with TensorFlow/Keras
- Predict which species a flower belongs to (Setosa, Versicolor, or Virginica)
- Visualizing accuracy every 25 epochs
### Species and their features:
####  Each sample includes:
- Sepal length
- Sepal width
- Petal length
- Petal width
#### The model learns from these four features to predict the species of the iris.

## Version History
### 1.0.0
- Terminal based
- Trains over 150 epochs
- Prints progress every 25 epochs
- Displays 5 predictions of 5 randomly selected flowers from the data set

## How to run 
1. Make sure python is installed

2. Clone the repository

3. Create and activate a virtual environment
```
python -m venv venv
venv\Scripts\activate   # on Windows
```
```
python3 -m venv venv
source venv/bin/activate # on macOS/Linux
```

4. Install external library
```
pip install pip install tensorflow pandas scikit-learn matplotlib
```
#### This will take a decent amount of time 

5. Run the program
```
python iris_classifier.py
```

## How to tweak the project for your own use 
1. Fork the repository
   
2. Clone the fork
   
3. Make your changes to the code
   
4. Commit and push your changes to the fork
   
5. OPTIONAL - Create a pull request if you want the main repository to change the code with what you changed

## How it works 
#### 1. Loads the iris dataset
  - This is a built in data set with 150 flower measurements

#### 2. Builds a neural network 
  - 4 input neurons (one for each measurement)
  - 16 neurons in the first hidden layer connected to the 4 inputs
  - 8 neurons in the second hidden layerconnected to the 16 neurons in the first layer
  - 3 output neurons (one for each species)

#### 3. Trains the model 
  - Goes through 150 epochs (one epoch is a runthrough of the data set)
  - Prints the accuracy every 25 epochs 
  - After every epoch, it gets a loss (how far off it was from the right answer) and adjusts its calculations

#### 4. Outputs a prediction for 5 random measurement sets

#### 5. Displays bar graphs and line charts to visualize predictions using matplotlib

## Outputs
```
Epoch   1: loss=1.1637, acc=0.2022, val_loss=1.1481, val_acc=0.3043
Epoch  25: loss=0.5382, acc=0.7640, val_loss=0.6579, val_acc=0.7826
Epoch  50: loss=0.3255, acc=0.8989, val_loss=0.4088, val_acc=0.9130
Epoch  75: loss=0.1915, acc=0.9551, val_loss=0.2585, val_acc=0.9565
Epoch 100: loss=0.1217, acc=0.9775, val_loss=0.1938, val_acc=0.9565
Epoch 125: loss=0.0884, acc=0.9775, val_loss=0.1663, val_acc=0.9565
Epoch 150: loss=0.0724, acc=0.9888, val_loss=0.1521, val_acc=0.9565

Final test accuracy: 0.921
```
![image](https://github.com/user-attachments/assets/74a1af7b-01ac-436b-a9d4-6e472d4a4881)

![image](https://github.com/user-attachments/assets/25030ba7-70a8-4830-9352-60844936ed01)

![image](https://github.com/user-attachments/assets/d75d4314-49ab-4ccb-b3ac-3509fc158400)

![image](https://github.com/user-attachments/assets/5d40cbcf-a278-4cef-b305-5f496e296c10)

![image](https://github.com/user-attachments/assets/d852a8ca-0b7c-4286-87ab-6095dd00df9a)

![image](https://github.com/user-attachments/assets/d31b3eba-d9c5-4f20-9ccc-5acae5ff3284)

![image](https://github.com/user-attachments/assets/3e2d7744-cc56-4520-84a4-9217ae7399ef)

![image](https://github.com/user-attachments/assets/92f72053-e374-48d9-84c9-a9d39d3e95a1)

```
Sample #1:
Measurements (scaled): Sepal L=2.33, Sepal W=-0.15, Petal L=1.34, Petal W=1.47
Actual species: 🌺 virginica
Predicted species: 🌺 virginica (confidence: 1.00)
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 24ms/step

Sample #2:
Measurements (scaled): Sepal L=-0.08, Sepal W=-0.83, Petal L=0.76, Petal W=0.93
Actual species: 🌺 virginica
Predicted species: 🌺 virginica (confidence: 0.98)
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 28ms/step

Sample #3:
Measurements (scaled): Sepal L=-0.34, Sepal W=-1.29, Petal L=0.07, Petal W=-0.13
Actual species: 🌷 versicolor
Predicted species: 🌷 versicolor (confidence: 0.99)
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 30ms/step

Sample #4:
Measurements (scaled): Sepal L=-0.21, Sepal W=-0.60, Petal L=0.19, Petal W=0.14
Actual species: 🌷 versicolor
Predicted species: 🌷 versicolor (confidence: 0.97)
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step

Sample #5:
Measurements (scaled): Sepal L=1.82, Sepal W=-0.37, Petal L=1.45, Petal W=0.80
Actual species: 🌺 virginica
Predicted species: 🌺 virginica (confidence: 0.97)
```

## Known Issues 
- Compatible with scikit-learn 1.2 and above







