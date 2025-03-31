import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import base64
import os
from io import BytesIO
from flask import Flask, render_template, request, jsonify

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Create Flask app with correct static folder configuration
app = Flask(__name__, static_folder='static', static_url_path='/static')

class LoanPredictionModel:
    def __init__(self, dataset_path):
        """
        Initialize the loan prediction model
        
        :param dataset_path: Path to the CSV file containing loan data
        """
        # Load the dataset
        self.loan_dataset = pd.read_csv(dataset_path)
        
        # Data preprocessing
        self.preprocess_data()
        
    def preprocess_data(self):
        """
        Preprocess the loan dataset by handling missing values 
        and encoding categorical variables
        """
        # Define expected columns
        expected_columns = [
            'Gender', 'Married', 'Dependents', 'Education', 
            'Self_Employed', 'Property_Area', 'Credit_History',
            'ApplicantIncome', 'CoapplicantIncome', 
            'LoanAmount', 'Loan_Amount_Term', 'Loan_Status'
        ]

        # Check for missing columns and add them with default values if not present
        for col in expected_columns:
            if col not in self.loan_dataset.columns:
                if col in ['Gender', 'Married', 'Dependents', 'Education', 
                           'Self_Employed', 'Property_Area']:
                    self.loan_dataset[col] = 'Unknown'
                elif col == 'Credit_History':
                    self.loan_dataset[col] = 0  # Default to 0
                elif col in ['ApplicantIncome', 'CoapplicantIncome', 
                             'LoanAmount', 'Loan_Amount_Term']:
                    self.loan_dataset[col] = 0
                elif col == 'Loan_Status':
                    self.loan_dataset[col] = 'N'

        # Handle missing values
        self.loan_dataset = self.loan_dataset.fillna({
            'Gender': 'Unknown',
            'Married': 'Unknown',
            'Dependents': 'Unknown',
            'Education': 'Unknown',
            'Self_Employed': 'Unknown',
            'Property_Area': 'Unknown',
            'Credit_History': 0,
            'ApplicantIncome': 0,
            'CoapplicantIncome': 0,
            'LoanAmount': 0,
            'Loan_Amount_Term': 0
        })
        
        # Identify column types
        categorical_columns = ['Gender', 'Married', 'Dependents', 
                               'Education', 'Self_Employed', 'Property_Area']
        numeric_columns = ['ApplicantIncome', 'CoapplicantIncome', 
                           'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
        
        # Label encoding for categorical columns
        label_encoder = LabelEncoder()
        for col in categorical_columns:
            self.loan_dataset[col] = label_encoder.fit_transform(
                self.loan_dataset[col].astype(str)
            )
        
        # Encode target variable
        self.loan_dataset['Loan_Status'] = label_encoder.fit_transform(
            self.loan_dataset['Loan_Status'].astype(str)
        )
        
        # Separate features and target
        self.X = self.loan_dataset.drop(columns=['Loan_ID', 'Loan_Status'], errors='ignore')
        self.Y = self.loan_dataset['Loan_Status']
        
        # Split the data
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X, self.Y, test_size=0.2, stratify=self.Y, random_state=42
        )
        
    def create_preprocessing_pipeline(self):
        """
        Create a preprocessing pipeline for the dataset
        
        :return: Preprocessor for handling numeric and categorical features
        """
        # Separate numeric and categorical columns
        numeric_features = self.X.select_dtypes(
            include=['int64', 'float64']
        ).columns
        categorical_features = self.X.select_dtypes(
            include=['int64']
        ).columns.difference(numeric_features)
        
        # Numeric transformer
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical transformer
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        return preprocessor
    
    def train_models(self):
        """
        Train multiple machine learning models with hyperparameter tuning
        
        :return: Dictionary of model results
        """
        # Create preprocessing pipeline
        preprocessor = self.create_preprocessing_pipeline()
        
        # Define models with their parameter grids
        models = {
            'SVM': {
                'model': Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', SVC())
                ]),
                'params': {
                    'classifier__C': [0.1, 1, 10],
                    'classifier__kernel': ['linear', 'rbf']
                }
            },
            'KNN': {
                'model': Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', KNeighborsClassifier())
                ]),
                'params': {
                    'classifier__n_neighbors': [3, 5, 7],
                    'classifier__weights': ['uniform', 'distance']
                }
            },
            'Random Forest': {
                'model': Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', RandomForestClassifier())
                ]), 
                'params': {
                    'classifier__n_estimators': [100, 200],
                    'classifier__max_depth': [None, 10, 20]
                }
            },
            'Logistic Regression': {
                'model': Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', LogisticRegression())
                ]),
                'params': {
                    'classifier__C': [0.1, 1, 10],
                    'classifier__penalty': ['l2']
                }
            }
        }
        
        # Train and evaluate models
        results = {}
        for name, model_info in models.items():
            # Perform grid search with cross-validation
            grid_search = GridSearchCV(
                model_info['model'], 
                model_info['params'], 
                cv=5, 
                scoring='accuracy',
                error_score='raise'
            )
            
            # Fit the model
            grid_search.fit(self.X_train, self.Y_train)
            
            # Store results
            results[name] = {
                'best_model': grid_search.best_estimator_,
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'test_score': grid_search.score(self.X_test, self.Y_test)
            }
            
            # Save the best model
            joblib.dump(grid_search.best_estimator_, f'{name}_loan_model.joblib')
        
        return results
    
    def visualize_results(self, results):
        """
        Visualize model performance results
        
        :param results: Dictionary of model results
        :return: Base64 encoded image and results dictionary
        """
        # Prepare data for visualization
        model_names = list(results.keys())
        train_scores = [results[name]['best_score'] for name in model_names]
        test_scores = [results[name]['test_score'] for name in model_names]
        
        # Create bar plot
        plt.figure(figsize=(10, 6))
        x = np.arange(len(model_names))
        width = 0.35
        
        plt.bar(x - width/2, train_scores, width, label='Cross-Validation Score')
        plt.bar(x + width/2, test_scores, width, label='Test Score')
        
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.title('Model Performance Comparison')
        plt.xticks(x, model_names, rotation=45)
        plt.legend()
        plt.tight_layout()
        
        # Save plot to a buffer
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        
        # Encode plot to base64 string
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return image_base64, results

# Create static directory if it doesn't exist
os.makedirs('static', exist_ok=True)

# Create static/style.css if it doesn't exist
if not os.path.exists('static/style.css'):
    with open('static/style.css', 'w') as f:
        f.write("""
body {
    background-color: #f5f5f5;
    font-family: 'Arial', sans-serif;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
}

header {
    background-color: #0d6efd;
    color: white;
    border-radius: 8px;
    margin-bottom: 20px;
}

.card {
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    border: none;
}

.card-header {
    background-color: #f8f9fa;
    font-weight: bold;
}

.btn-primary {
    background-color: #0d6efd;
    border: none;
}

.btn-primary:hover {
    background-color: #0b5ed7;
}
""")

# Create static/script.js if it doesn't exist
if not os.path.exists('static/script.js'):
    with open('static/script.js', 'w') as f:
        f.write("""
$(document).ready(function() {
    // Handle form submission
    $('#loan-form').on('submit', function(e) {
        e.preventDefault();
        
        // Show loading state
        $('#result-container').html('<div class="alert alert-info">Processing your application...</div>');
        
        // Send form data to server
        $.ajax({
            url: '/predict',
            method: 'POST',
            data: $(this).serialize(),
            success: function(response) {
                // Update result container
                $('#result-container').html(
                    '<div class="alert ' + response.class + '">' + response.message + '</div>'
                );
            },
            error: function() {
                // Handle error
                $('#result-container').html(
                    '<div class="alert alert-danger">Error processing your request. Please try again.</div>'
                );
            }
        });
    });
    
    // Handle train button click
    $('#train-button').on('click', function() {
        // Show loading state
        $(this).html('Training...').prop('disabled', true);
        
        // Send request to train models
        $.ajax({
            url: '/train',
            method: 'GET',
            success: function(response) {
                if (response.status === 'success') {
                    // Redirect to main page
                    window.location.href = response.redirect;
                } else {
                    // Show error
                    alert('Error: ' + response.message);
                    $('#train-button').html('Train Models').prop('disabled', false);
                }
            },
            error: function() {
                // Handle error
                alert('Error training models. Please try again.');
                $('#train-button').html('Train Models').prop('disabled', false);
            }
        });
    });
});
""")

# Initialize predictor and train models on app startup
model_path = "train_u6lujuX_CVtuZ9i (1).csv"  # Update with your dataset path

# Create a sample dataset if the actual one doesn't exist
if not os.path.exists(model_path):
    print(f"Creating sample dataset at {model_path}")
    sample_data = pd.DataFrame({
        'Loan_ID': [f'LOAN{i:04d}' for i in range(100)],
        'Gender': np.random.choice(['Male', 'Female'], 100),
        'Married': np.random.choice(['Yes', 'No'], 100),
        'Dependents': np.random.choice(['0', '1', '2', '3+'], 100),
        'Education': np.random.choice(['Graduate', 'Not Graduate'], 100),
        'Self_Employed': np.random.choice(['Yes', 'No'], 100),
        'ApplicantIncome': np.random.randint(2000, 10000, 100),
        'CoapplicantIncome': np.random.randint(0, 5000, 100),
        'LoanAmount': np.random.randint(50, 500, 100),
        'Loan_Amount_Term': np.random.choice([360, 180, 240, 120], 100),
        'Credit_History': np.random.choice([1, 0], 100),
        'Property_Area': np.random.choice(['Urban', 'Rural', 'Semiurban'], 100),
        'Loan_Status': np.random.choice(['Y', 'N'], 100)
    })
    sample_data.to_csv(model_path, index=False)

loan_predictor = None
model_results = None
chart_image = None

@app.route('/')
def index():
    global chart_image, model_results
    return render_template('index.html', chart_image=chart_image, results=model_results)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        form_data = {
            'Gender': request.form.get('gender'),
            'Married': request.form.get('married'),
            'Dependents': request.form.get('dependents'),
            'Education': request.form.get('education'),
            'Self_Employed': request.form.get('self_employed'),
            'Property_Area': request.form.get('property_area'),
            'ApplicantIncome': float(request.form.get('applicant_income', 0)),
            'CoapplicantIncome': float(request.form.get('coapplicant_income', 0)),
            'LoanAmount': float(request.form.get('loan_amount', 0)),
            'Loan_Amount_Term': float(request.form.get('loan_amount_term', 0)),
            'Credit_History': float(request.form.get('credit_history', 0)),
            'MonthlyDebt': float(request.form.get('monthly_debt', 0)),  # New field for DTI calculation
            'EmploymentYears': float(request.form.get('employment_years', 0))  # New field for employment history
        }

        # Convert to DataFrame
        df = pd.DataFrame([form_data])

        # Apply category mappings
        categorical_mappings = {
            'Gender': {'Male': 1, 'Female': 0},
            'Married': {'Yes': 1, 'No': 0},
            'Dependents': {'0': 0, '1': 1, '2': 2, '3+': 3},
            'Education': {'Graduate': 1, 'Not Graduate': 0},
            'Self_Employed': {'Yes': 1, 'No': 0},
            'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2}
        }

        # Apply mappings
        for col, mapping in categorical_mappings.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)

        # Enhanced loan eligibility check based on the specified criteria
        credit_score = float(request.form.get('credit_history', 0))
        loan_amount = form_data.get('LoanAmount', 0)
        applicant_income = form_data.get('ApplicantIncome', 0)
        coapplicant_income = form_data.get('CoapplicantIncome', 0)
        monthly_debt = form_data.get('MonthlyDebt', 0)
        employment_years = form_data.get('EmploymentYears', 0)
        total_income = applicant_income + coapplicant_income
        monthly_income = total_income / 12
        dependents = form_data.get('Dependents', '0')
        is_graduate = form_data.get('Education') == 'Graduate'
        is_married = form_data.get('Married') == 'Yes'
        is_self_employed = form_data.get('Self_Employed') == 'Yes'
        
        # Convert dependents to integer
        if dependents == '3+':
            num_dependents = 3
        else:
            num_dependents = int(dependents) if dependents.isdigit() else 0
        
        # Calculate DTI (Debt-to-Income Ratio)
        dti = monthly_debt / monthly_income if monthly_income > 0 else 1.0
        
        # Calculate Loan-to-Income Ratio
        loan_to_income_ratio = loan_amount / total_income if total_income > 0 else float('inf')
        
        # Initialize eligibility factors
        eligibility_factors = []
        rejection_factors = []
        
        # 1. Credit Score Check (as per criteria)
        if credit_score >= 700:
            eligibility_factors.append("Excellent credit score")
        elif 600 <= credit_score < 700:
            eligibility_factors.append("Average credit score")
        else:
            rejection_factors.append("Low credit score")
            
        # 2. DTI Ratio Check
        if dti < 0.4:
            eligibility_factors.append("Healthy debt-to-income ratio")
        else:
            rejection_factors.append("High debt-to-income ratio")
            
        # 3. Loan Amount to Income Check
        if loan_to_income_ratio < 3:
            eligibility_factors.append("Loan amount is reasonable compared to income")
        elif 3 <= loan_to_income_ratio < 6:
            eligibility_factors.append("Loan amount is moderately high compared to income")
        else:
            rejection_factors.append("Loan amount too high compared to income")
            
        # 4. Dependents Consideration
        if num_dependents >= 3 and dti > 0.35:
            rejection_factors.append("High financial obligations due to dependents")
        
        # 5. Employment History
        if employment_years >= 2:
            eligibility_factors.append("Stable employment history")
        else:
            eligibility_factors.append("Limited employment history")
            
        # 6. Education Factor
        if is_graduate:
            eligibility_factors.append("Higher education qualification")
        
        # 7. Combined Income for Married Applicants
        if is_married and coapplicant_income > 0:
            eligibility_factors.append("Dual income household")
            
        # 8. Self Employment Factor
        if is_self_employed and employment_years < 2:
            rejection_factors.append("Recently self-employed (income stability concern)")
        
        # Make preliminary decision based on criteria
        if credit_score < 600:
            return jsonify({
                'status': 'rejected',
                'message': 'Loan Rejected: Low Credit Score (Below 600)',
                'class': 'alert-danger'
            })
            
        if dti > 0.6:
            return jsonify({
                'status': 'rejected',
                'message': 'Loan Rejected: Debt-to-Income Ratio Too High',
                'class': 'alert-danger'
            })
            
        if loan_to_income_ratio > 10:
            return jsonify({
                'status': 'rejected',
                'message': 'Loan Rejected: Loan Amount Too High Compared to Income',
                'class': 'alert-danger'
            })
            
        # Special case for self-employed with recent business history
        if is_self_employed and employment_years < 1.5:
            return jsonify({
                'status': 'pending',
                'message': 'Loan Pending: Additional Income Verification Required for Self-Employed',
                'class': 'alert-warning'
            })
            
        # Check if model exists, if not train it
        if not os.path.exists('SVM_loan_model.joblib'):
            return jsonify({
                'status': 'error',
                'message': 'Model not trained yet. Please train the model first.',
                'class': 'alert-warning'
            })

        # Load the best model
        best_model = joblib.load('SVM_loan_model.joblib')
        
        # Prepare data for prediction by removing columns not in training data
        prediction_df = df.drop(columns=['MonthlyDebt', 'EmploymentYears'], errors='ignore')

        # Predict using the model
        prediction = best_model.predict(prediction_df)

        # Final Decision
        if prediction[0] == 1 and not rejection_factors:
            if 600 <= credit_score < 700 or dti > 0.4 or loan_to_income_ratio > 5:
                # Conditional approval with higher interest
                factor_text = ", ".join(eligibility_factors)
                constraint_text = "May require higher interest rate"
                return jsonify({
                    'status': 'conditional',
                    'message': f'Conditionally Approved: {constraint_text}. Positive factors: {factor_text}',
                    'class': 'alert-info'
                })
            else:
                # Full approval
                factor_text = ", ".join(eligibility_factors[:3])  # Show top 3 factors
                return jsonify({
                    'status': 'approved',
                    'message': f'Congratulations! Loan Approved. Positive factors: {factor_text}',
                    'class': 'alert-success'
                })
        elif prediction[0] == 1 and len(rejection_factors) < 2:
            # Conditionally approved but with concerns
            rejection_text = ", ".join(rejection_factors)
            return jsonify({
                'status': 'conditional',
                'message': f'Conditionally Approved with Concerns: {rejection_text}',
                'class': 'alert-info'
            })
        else:
            # Rejected
            rejection_text = ", ".join(rejection_factors[:2])  # Top 2 rejection reasons
            return jsonify({
                'status': 'rejected',
                'message': f'Sorry, Loan Not Approved. Reasons: {rejection_text}',
                'class': 'alert-danger'
            })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error: {str(e)}',
            'class': 'alert-danger'
        })

@app.route('/train', methods=['GET'])
def train():
    global loan_predictor, model_results, chart_image
    try:
        # Train models
        loan_predictor = LoanPredictionModel(model_path)
        model_results = loan_predictor.train_models()
        chart_image, model_results = loan_predictor.visualize_results(model_results)
        
        # Remove model objects from results (not JSON serializable)
        for model in model_results:
            model_results[model].pop('best_model', None)
        
        return jsonify({
            'status': 'success',
            'message': 'Models trained successfully',
            'redirect': '/'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error training models: {str(e)}'
        })

if __name__ == "__main__":
    # Train models on startup
    try:
        loan_predictor = LoanPredictionModel(model_path)
        model_results = loan_predictor.train_models()
        chart_image, model_results = loan_predictor.visualize_results(model_results)
        
        # Remove model objects from results (not JSON serializable)
        for model in model_results:
            model_results[model].pop('best_model', None)
    except Exception as e:
        print(f"Error during initial training: {str(e)}")
        chart_image = None
        model_results = None
        
    app.run(debug=True)