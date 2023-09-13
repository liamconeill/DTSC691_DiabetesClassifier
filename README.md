![CapReadme.png](attachment:CapReadme.png)

# Why did I make this?

Health is wealth, and one of the largest preventable health conditions plaguing the population today is diabetes. So I decided to build a tool to help screen a person's risk.

This project was created for my applied machine learning capstone project and I chose to train a binary classifier which predicts a user's risk for diabetes and prediabetes based on a short survey of relevant health behaviours.

With the project notebook you can follow all the steps used for the project:

- Background Information
- Data Importing
- Data Cleaning
- Exploratory Data Analysis
- Model Selection
- Hyperparameter Tuning
- Model Evaluation

You can try the application yourself by following the steps listed under "Flask Application". When you navigate to the survey page, input your answers to the questions, and click the predict button!

![FlaskDemo.gif](attachment:FlaskDemo.gif)

## Data Source

The data used for this project was the 2020 data from the Behavioral Risk Factor Surveillance System (BRFSS) conducted yearly by the Centers for Disease Control and Prevention (CDC). You can find a description of the survey as well as yearly data at <a href="https://www.cdc.gov/brfss/index.html">https://www.cdc.gov/brfss/index.html</a>.

The .csv file used was a cleaned version taken from Kaggle user Ahmet Emre found <a href="https://www.kaggle.com/datasets/aemreusta/brfss-2020-survey-data">here</a>.

## Flask Application

To run the application for yourself, download the files included in the folder "Flask Web App".

Once you have the files, navigate to their directory in your terminal and enter the following command:


```python
python server.py
```

## Dependencies

This project was created with the following technologies and versions:

- Flask (v2.2.2)
- Joblib (v1.1.1)
- Keras (v2.10.0)
- Matplotlib (v3.6.2)
- Numpy (v1.23.5)
- Pandas (v1.5.3)
- Python (v3.9.13)
- Scikit-learn (v1.2.2)
- Seaborn (v0.12.2)
- Tensorflow (v2.10.0)
- XGBoost (v1.7.3)

If you don't currently have the dependencies installed, and you aren't familiar with installing packages using pip, I would recommend using <a href="https://www.anaconda.com/download">Anaconda</a> to install these as it helps manage the packages and their own dependencies, as well as let you create a virtual environment for the specific requirements.

#### Thanks for taking the time to look at my project. If you have any questions you can reach me at:
- liamcamerononeill@gmail.com
- <a href="https://www.linkedin.com/in/liamconeill/">LinkedIn</a>


```python

```
