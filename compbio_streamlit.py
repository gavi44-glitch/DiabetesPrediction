import pickle
import pandas as pd
import gdown 
import streamlit as st 
import os 
import matplotlib.pyplot as plt 

# proses load model training dan scaler 
url = 'https://drive.google.com/uc?id=1Ra1GCSayL-mav3vmLKQE2Rac4lq3xXnw'
gdown.download(url, 'random_forest_model.pkl', quiet=False)

with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

url2 = 'https://drive.google.com/uc?id=1mOlIHsRagoPmlj6x5B2XJHkmUCZtu0bG'

gdown.download(url2, 'scaler.pkl', quiet=False)

with open('scaler.pkl', 'rb') as f2:
    scaler = pickle.load(f2)


st.set_page_config(page_title='Prediksi Diabetes', layout = 'centered')

file_id = "1ezWNawXq0Gfm_SO1IZJm9OwojKb7lt90"
url = f"https://drive.google.com/uc?id={file_id}"

output = "riwayat_input.csv"
gdown.download(url, output, quiet=False)


if os.path.exists(output):
    history = pd.read_csv(output).to_dict(orient="records")
else:
    history = []

st.title('Prediksi Diabetes')

st.markdown('Input fitur berikut pada pasien : ')
input_data = {}
col1, col2 = st.columns(2)

with col1:
    input_data['Pregnancies'] = st.number_input("Pregnancies", min_value=0, value=2)
    input_data['Glucose'] = st.number_input("Glucose", min_value=0.0, value=155.0)
    input_data['BloodPressure'] = st.number_input("BloodPressure", min_value=0.0, value=74.0)
    input_data['SkinThickness'] = st.number_input("SkinThickness", min_value=0.0, value=22.0)

with col2:
    input_data['Insulin'] = st.number_input("Insulin", min_value=0.0, value=340.0)
    input_data['BMI'] = st.number_input("BMI", min_value=0.0, value=30.7)
    input_data['DiabetesPedigreeFunction'] = st.number_input("DiabetesPedigreeFunction", min_value=0.0, value=0.388)
    input_data['Age'] = st.number_input("Age", min_value=0, value=45)

if st.button('Prediksi Diabetes') : 
    def BMI_category(bmi):
        if bmi < 18.5:
            return 'Underweight'
        elif 18.5 <= bmi <= 24.9:
            return 'Normal'
        elif 24.9 < bmi <= 29.9:
            return 'Overweight'
        elif 29.9 < bmi <= 34.9:
            return 'Obesity 1'
        elif 34.9 < bmi <= 39.9:
            return 'Obesity 2'
        else:
            return 'Obesity 3'

    def insulin_category(insulin):
        if 16 <= insulin <= 166:
            return 'Normal'
        else:
            return 'Abnormal'

    def glucose_category(glucose):
        if glucose <= 70:
            return 'Low'
        elif 70 < glucose <= 99:
            return 'Normal'
        elif 99 < glucose <= 126:
            return 'Overweight'
        else:
            return 'Secret'

    df = pd.DataFrame([input_data])

    df['NewBMI'] = df['BMI'].apply(BMI_category)
    df['NewInsulinScore'] = df['Insulin'].apply(insulin_category)
    df['NewGlucose'] = df['Glucose'].apply(glucose_category)

    df = pd.get_dummies(df, columns=['NewBMI', 'NewInsulinScore', 'NewGlucose'])

    expected_columns = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
        'BMI', 'DiabetesPedigreeFunction', 'Age',
        'NewBMI_Obesity 1', 'NewBMI_Obesity 2', 'NewBMI_Obesity 3',
        'NewBMI_Overweight', 'NewBMI_Underweight',
        'NewInsulinScore_Normal',
        'NewGlucose_Low', 'NewGlucose_Normal', 'NewGlucose_Overweight', 'NewGlucose_Secret',
    ]

    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[expected_columns]

    # prediksi
    data_scaled = scaler.transform(df)
    prediction = model.predict(data_scaled)[0]
    prob = model.predict_proba(data_scaled)[0][1]

    # output hasil
    if prediction == 1:
        st.error("🔴 Hasil Prediksi: **Diabetes**")
    else:
        st.success("🟢 Hasil Prediksi: **Tidak Diabetes**")

    st.write(f"Probabilitas diabetes (kelas 1): **{prob:.2f}**")
    st.write(f"Probabilitas tidak diabetes (kelas 0): **{1 - prob:.2f}**")

    # untuk save ke file csv
    record = input_data.copy()
    record['Prediction'] = 'Diabetes' if prediction == 1 else 'Tidak Diabetes'
    record['Probability'] = prob
    history.append(record)

    pd.DataFrame(history).to_csv(output, index=False)

# untuk tampilkan grafik perubahan gula dan insulin 
if len(history) > 1:
    hist_df = pd.DataFrame(history)

    st.subheader("📈 Riwayat Glucose dan Insulin")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(hist_df['Glucose'], marker='o', label='Glucose')
    ax1.set_title("Perubahan Kadar Glukosa")
    ax1.set_xlabel("Percobaan ke-")
    ax1.set_ylabel("Glucose")
    ax1.grid(True)

    ax2.plot(hist_df['Insulin'], marker='o', color='orange', label='Insulin')
    ax2.set_title("Perubahan Kadar Insulin")
    ax2.set_xlabel("Percobaan ke-")
    ax2.set_ylabel("Insulin")
    ax2.grid(True)

    st.pyplot(fig)





