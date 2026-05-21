# Customer Churn Prediction using ANN

## Overview
This project predicts customer churn using an Artificial Neural Network (ANN) built with TensorFlow/Keras.

The objective is to classify whether a customer is likely to leave the bank based on customer attributes such as credit score, geography, gender, balance, salary, and account activity.

This project focuses on:
- Data preprocessing
- Feature engineering
- One-hot encoding
- Feature scaling
- ANN architecture design
- Binary classification
- Model evaluation

---

## Dataset
Dataset used:
- Credit Card Customer Churn Prediction Dataset

Features include:
- CreditScore
- Geography
- Gender
- Age
- Tenure
- Balance
- NumOfProducts
- HasCrCard
- IsActiveMember
- EstimatedSalary

Target variable:
- Exited (0 = Stayed, 1 = Churned)

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- TensorFlow
- Keras

---

## Data Preprocessing

The preprocessing pipeline includes:
- Removing unnecessary columns
- One-hot encoding categorical variables
- Avoiding dummy variable trap using `drop_first=True`
- Train-test split
- Feature scaling using StandardScaler

---

## ANN Architecture

The neural network architecture:

Input Layer → Hidden Layer → Output Layer

### Model Structure

```python
model = Sequential()

model.add(Input(shape=(11,)))
model.add(Dense(11, activation='relu'))
model.add(Dense(11, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

### Activation Functions
- ReLU for hidden layers
- Sigmoid for output layer

Sigmoid output:

```math
\hat{y} = \frac{1}{1 + e^{-z}}
```

---

## Model Compilation

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

Loss function:
- Binary Cross Entropy

Optimizer:
- Adam Optimizer

---

## Training

```python
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=100
)
```

---

## Evaluation

The project includes:
- Training accuracy visualization
- Validation accuracy visualization
- Training loss visualization
- Validation loss visualization

These graphs help analyze:
- Learning behavior
- Overfitting
- Model convergence

---

## Future Improvements

Possible future enhancements:
- Dropout regularization
- Hyperparameter tuning
- Early stopping
- Batch normalization
- PyTorch implementation
- Advanced ANN architectures

---

## Author

Atul Kumar  
B.Tech – Artificial Intelligence & Data Science  
IIITDM Kurnool
