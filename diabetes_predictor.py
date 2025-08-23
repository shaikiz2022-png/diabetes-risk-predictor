import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load dataset
df = pd.read_csv("diabetes.csv")

# Features and labels
X = df.drop("Outcome", axis=1).values
y = df["Outcome"].values

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Define model
class DiabetesModel(nn.Module):
    def __init__(self):
        print("Entering _init_ of DiabetesModel at", pd.Timestamp.now())
        super().__init__()
        self.layer1 = nn.Linear(8, 16)
        self.layer2 = nn.Linear(16, 8)
        self.output = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        print("Layers initialized:", [self.layer1, self.layer2, self.output])
        params = list(self.parameters())
        print("Parameters after init:", params)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.output(x))
        return x

print("Creating model instance at", pd.Timestamp.now())
model = DiabetesModel()
print("Model instance created, parameters:", list(model.parameters()))

# Loss and optimizer
print("Setting up optimizer at", pd.Timestamp.now())
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
print("Optimizer set up, parameters:", list(model.parameters()))

# Training loop
epochs = 500
for epoch in range(epochs):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Evaluate model
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_classes = (y_pred > 0.5).float()
    accuracy = (y_pred_classes.eq(y_test).sum() / float(y_test.shape[0])).item()
    print(f"Accuracy: {accuracy * 100:.2f}%")

# Function to predict diabetes for new input
def predict_diabetes(input_data):
    input_np = np.array([input_data])
    scaled = scaler.transform(input_np)
    input_tensor = torch.tensor(scaled, dtype=torch.float32)
    with torch.no_grad():
        prediction = model(input_tensor)
        return "Diabetic" if prediction.item() > 0.5 else "Not Diabetic", prediction.item()

# Streamlit interface
st.title("Diabetes Risk Predictor")
st.markdown("""
Enter the following details to predict the likelihood of diabetes. All fields are required.
Values should be based on medical measurements. If unsure, consult a healthcare professional.
""")

# Organize inputs in two columns
col1, col2 = st.columns(2)

# Input fields with labels and help text
with col1:
    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0.0,
        max_value=20.0,
        value=0.0,
        step=1.0,
        help="Number of times pregnant (e.g., 0 if never pregnant)."
    )
    glucose = st.number_input(
        "Glucose",
        min_value=0.0,
        max_value=300.0,
        value=100.0,
        step=1.0,
        help="Plasma glucose concentration (mg/dL, typically 0–200)."
    )
    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0.0,
        max_value=200.0,
        value=70.0,
        step=1.0,
        help="Diastolic blood pressure (mm Hg, typically 0–120)."
    )
    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0.0,
        max_value=100.0,
        value=20.0,
        step=1.0,
        help="Triceps skinfold thickness (mm, typically 0–50)."
    )

with col2:
    insulin = st.number_input(
        "Insulin",
        min_value=0.0,
        max_value=1000.0,
        value=100.0,
        step=1.0,
        help="2-hour serum insulin (mu U/ml, typically 0–300)."
    )
    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=100.0,
        value=25.0,
        step=0.1,
        help="Body Mass Index (weight in kg/(height in m)^2, typically 15–50)."
    )
    dpf = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.5,
        step=0.01,
        help="A score indicating genetic predisposition to diabetes (typically 0–2.5)."
    )
    age = st.number_input(
        "Age",
        min_value=0.0,
        max_value=120.0,
        value=30.0,
        step=1.0,
        help="Age in years."
    )

# Button to predict
if st.button("Predict Diabetes Risk"):
    input_list = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
    # Validate inputs
    if any(np.isnan(x) or x is None for x in input_list):
        st.error("Please fill in all fields with valid numbers.")
    elif glucose == 0 or blood_pressure == 0 or bmi == 0:
        st.warning("Warning: Glucose, Blood Pressure, or BMI values of 0 are unrealistic. Please verify your inputs.")
    else:
        with st.spinner("Calculating prediction..."):
            result, probability = predict_diabetes(input_list)
            st.success(f"*Prediction: {result}* (Probability: {probability:.2%})")
            if result == "Diabetic":
                st.warning("This is a predictive model. Please consult a healthcare professional for a proper diagnosis.")
            else:
                st.info("This is a predictive model. Maintain a healthy lifestyle to reduce risk.")

# Reset button
if st.button("Reset Inputs"):
    st.rerun()

# Example input section
with st.expander("See Example Input"):
    st.write("""
    Example values (based on typical dataset ranges):
    - Pregnancies: 2
    - Glucose: 120
    - Blood Pressure: 70
    - Skin Thickness: 20
    - Insulin: 80
    - BMI: 32
    - Diabetes Pedigree Function: 0.47
    - Age: 29
    """)

# Console input (only runs if script is executed directly, not via Streamlit)
    print("\nEnter patient details:")
    preg = float(input("Pregnancies: "))
    glu = float(input("Glucose: "))
    bp = float(input("BloodPressure: "))
    skin = float(input("SkinThickness: "))
    ins = float(input("Insulin: "))
    bmi = float(input("BMI: "))
    dpf = float(input("DiabetesPedigreeFunction: "))
    age = float(input("Age: "))

    user_input = [preg, glu, bp, skin, ins, bmi, dpf, age]
    result = predict_diabetes(user_input)
    print(f"\nPrediction: {result} (probability: {probability:.2%})")