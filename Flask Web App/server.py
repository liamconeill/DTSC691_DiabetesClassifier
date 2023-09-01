from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb

app = Flask(__name__)

# Load the model
model = joblib.load('model.joblib')

# Load the scaler
scaler = joblib.load('scaler.joblib')

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Predict page
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        PHYSHLTH = request.form['PHYSHLTH']
        MENTHLTH = request.form['MENTHLTH']
        SLEEP = request.form['SLEEP']
        AGE = request.form['AGE']
        BMI = request.form['BMI']
        ARTHRITIS = request.form['ARTHRITIS']
        ASTHMA = request.form['ASTHMA']
        EXERCISE = request.form['EXERCISE']
        HEART_DISEASE = request.form['HEART_DISEASE']
        KIDNEY = request.form['KIDNEY']
        DEPRESSION = request.form['DEPRESSION']
        HLTHPLN1 = request.form['HLTHPLN1']
        SEX = request.form['SEX']
        SMOKER = request.form['SMOKER']
        STROKE = request.form['STROKE']
        WALK_ISSUE = request.form['WALK_ISSUE']
        DRINK = request.form['DRINK']
        GENHLTH = request.form['GENHLTH']
        ETHNICITY = request.form['ETHNICITY']
        EDUCATION = request.form['EDUCATION']
    
        feature_names = ['PHYSHLTH', 'MENTHLTH', 'SLEEP', 'AGE', 'BMI', 'ARTHRITIS', 'ASTHMA', 'EXERCISE',
                        'HEART_DISEASE', 'KIDNEY', 'DEPRESSION', 'HLTHPLN1', 'SEX', 'SMOKER',
                        'STROKE', 'WALK_ISSUE', 'DRINK', 'GENHLTH', 'ETHNICITY', 'EDUCATION']

        data = pd.DataFrame([pd.Series([PHYSHLTH, MENTHLTH, SLEEP, AGE, BMI, ARTHRITIS, ASTHMA, EXERCISE,
                                        HEART_DISEASE, KIDNEY, DEPRESSION, HLTHPLN1, SEX, SMOKER,
                                        STROKE, WALK_ISSUE, DRINK, GENHLTH, ETHNICITY, EDUCATION],
                                    index=feature_names)])
        
        # Perform preprocessing steps
        numerical_cols = ['PHYSHLTH', 'MENTHLTH', 'SLEEP', 'AGE', 'BMI']
        numerical = data[numerical_cols].astype(float).copy()
        binary_cols = ['ARTHRITIS', 'ASTHMA', 'EXERCISE', 'HEART_DISEASE', 'KIDNEY', 'DEPRESSION', 'HLTHPLN1', 'SEX', 'SMOKER', 'STROKE',
                       'WALK_ISSUE', 'DRINK']
        binary = data[binary_cols].astype(int).copy()
        categorical_cols = ['GENHLTH', 'ETHNICITY', 'EDUCATION']
        categorical = data[categorical_cols].astype(float).copy()

        # Perform one-hot encoding on categorical data
        all_categorical_values = {
            'GENHLTH': [1.0, 2.0, 3.0, 4.0, 5.0],
            'ETHNICITY': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'EDUCATION': [1.0, 2.0, 3.0, 4.0]
        }

        # Perform one-hot encoding on categorical data
        categorical_dummies = pd.get_dummies(categorical, columns=categorical_cols, 
                                            drop_first=False, dummy_na=False)

        # Add missing categorical columns with values set to 0
        for feature, values in all_categorical_values.items():
            missing_cols = set(values) - set(categorical.columns)
            for value in missing_cols:
                categorical_dummies[f'{feature}_{value}'] = 0

        training_categorical_cols = ['GENHLTH_1.0', 'GENHLTH_2.0', 'GENHLTH_3.0', 'GENHLTH_4.0', 'GENHLTH_5.0', 'ETHNICITY_1.0', 'ETHNICITY_2.0',
                                    'ETHNICITY_3.0', 'ETHNICITY_4.0', 'ETHNICITY_5.0', 'ETHNICITY_6.0', 'EDUCATION_1.0', 'EDUCATION_2.0', 
                                    'EDUCATION_3.0', 'EDUCATION_4.0']

        # Reorder columns to match the training data
        categorical_dummies = categorical_dummies.reindex(columns=training_categorical_cols, 
                                                          fill_value=0)

        # Log transform numerical values
        numerical = np.log1p(numerical)

        # Combine the preprocessed features
        features_transformed = pd.concat([numerical, binary, categorical_dummies], axis=1)

        # Scale the numerical data
        features_transformed = scaler.transform(features_transformed)

        prediction=model.predict_proba(features_transformed) ## Predicting the output
        output='{0:.{1}f}'.format(prediction[0][1], 2) ## Formating output

        if output>str(0.5):
            return render_template('predict.html',pred='You are likely to develop diabetes. You should consult a healthcare professional.')
        else:
            return render_template('predict.html',pred='You are unlikely to develop diabetes. You should still educate yourself on ideal health behaviours to protect yourself.')
        
    # GET request or form not submitted yet
    return render_template('predict.html', pred='')

if __name__ == '__main__':
    app.run(debug=True)