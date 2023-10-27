# Heart Disease Prediction with Logistic Regression

This repository contains a simple Python script for predicting heart disease using logistic regression. We utilize the popular libraries `pandas`, `numpy`, and `scikit-learn` for data manipulation, analysis, and machine learning. The script loads a heart disease dataset, trains a logistic regression model, and makes predictions based on user-provided input data.

## Prerequisites

Before running the script, ensure that you have the following libraries installed:

- `pandas`
- `numpy`
- `scikit-learn`

You can install these libraries using `pip`:

```bash
pip install pandas numpy scikit-learn
```

## Getting Started

1. Clone this repository to your local machine or download the script file.

2. Make sure to have your dataset in a CSV file format. In this example, we assume the dataset is named `heart.csv`.

3. Open the Python script and make sure the file path for your dataset is correctly specified:

   ```python
   df = pd.read_csv("/path/to/your/heart.csv")
   ```

## Running the Script

You can run the script by executing it using your Python interpreter. The script performs the following steps:

1. Load the heart disease dataset.

2. Split the data into training and testing sets using `train_test_split`.

3. Train a logistic regression model on the training data.

4. Calculate the accuracy of the model on both the training and testing data.

5. Make predictions based on user-provided input data.

6. Display whether the person has a heart disease or not.

To run the script, simply execute it in your preferred Python environment:

```bash
python heart_disease_prediction.py
```

## Customizing Input Data

You can customize the input data that you want to predict by modifying the `input_data` variable in the script. This variable should be a tuple containing 13 features related to heart health, such as age, sex, cholesterol levels, and more. For example:

```python
input_data = (18, 1, 2, 112, 230, 0, 0, 165, 0, 2.5, 1, 1, 3)
```

You can change these values to make predictions for different individuals.

## Results

The script will print out the prediction for the input data and whether the person is predicted to have heart disease or not, based on the trained logistic regression model. The script also prints the accuracy of the model on the training and testing data.

## Acknowledgments

This script is for educational purposes and serves as a simple example of using logistic regression for heart disease prediction. The accuracy mentioned in the script may vary based on the dataset and preprocessing.

Feel free to explore and modify the script to improve its performance or adapt it for your specific use case.

**Note:** Always consult with a medical professional for accurate heart disease diagnosis and treatment. The script is not a substitute for medical advice.

  these are some screenshots of the model
  https://github.com/Aditya9598/Heart-attack-prediction-model-using-linear-regression/blob/main/img2.jpg
  https://github.com/Aditya9598/Heart-attack-prediction-model-using-linear-regression/blob/main/img1.jpg
