import os
import pickle
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)


try:
    working_dir = os.path.dirname(os.path.abspath(__file__))
    diabetes_model = pickle.load(open(os.path.join(working_dir, 'diabetes_svm_model.sav'), 'rb'))
    heart_disease_model = pickle.load(open(os.path.join(working_dir, 'heart_disease_model1.sav'), 'rb'))
    parkinsons_model = pickle.load(open(os.path.join(working_dir, 'parkinsons_model.sav'), 'rb'))
except Exception as e:
    print(f"Error loading models: {e}")

# Home page route
@app.route('/')
def home():
    return render_template('index.html')

# Diabetes prediction route
@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    try:
        data = [float(request.form[key]) for key in [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
            'BMI', 'DiabetesPedigreeFunction', 'Age']]
        prediction = diabetes_model.predict([data])[0]
        result = 'The person is diabetic' if prediction == 1 else 'The person is not diabetic'
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})

# Heart disease prediction route
@app.route('/predict_heart_disease', methods=['POST'])
def predict_heart_disease():
    try:
        data = [float(request.form[key]) for key in [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
        prediction = heart_disease_model.predict([data])[0]
        result = 'The person has heart disease' if prediction == 1 else 'The person does not have heart disease'
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})

# Parkinson's disease prediction route
@app.route('/predict_parkinsons', methods=['POST'])
def predict_parkinsons():
    try:
        data = [float(request.form[key]) for key in [
            'fo', 'fhi', 'flo', 'Jitter_percent', 'Jitter_Abs',
            'RAP', 'PPQ', 'DDP', 'Shimmer', 'Shimmer_dB',
            'APQ3', 'APQ5', 'APQ', 'DDA', 'NHR',
            'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']]
        prediction = parkinsons_model.predict([data])[0]
        result = "The person has Parkinson's disease" if prediction == 1 else "The person does not have Parkinson's disease"
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
