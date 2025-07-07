# predict.py
import sys
import joblib

# Load model
model = joblib.load('spam_model.pkl')

# Get input from command line
if len(sys.argv) < 2:
    print("Usage: python predict.py \"your message here\"")
    sys.exit()

input_text = sys.argv[1]

# Predict
pred = model.predict([input_text])[0]
label = "SPAM" if pred == 1 else "NOT SPAM"

print(f"Prediction: {label}")