import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter  # For loading TFLite models
import numpy as np  # For data manipulation, especially with TFLite models

# Function to retrieve data from Xano and save it as a CSV file
def retrieve_data():
    xano_api_endpoint_bg = 'https://x8ki-letl-twmt.n7.xano.io/api:U4wk_Gn6/BackgroundReading'
    payload_bg = {}
    response_bg = requests.get(xano_api_endpoint_bg, params=payload_bg)

    if response_bg.status_code == 200:
        data_bg = response_bg.json()
    else:
        error_message = "Failed to retrieve data. Status code: " + str(response_bg.status_code)
        st.error(error_message)
        return None

    xano_api_endpoint_spectral = 'https://x8ki-letl-twmt.n7.xano.io/api:DKaWNKM4/spectral_data'
    payload_spectral = {}
    response_spectral = requests.get(xano_api_endpoint_spectral, params=payload_spectral)

    if response_spectral.status_code == 200:
        data_spectral = response_spectral.json()
    else:
        error_message = "Failed to retrieve data. Status code: " + str(response_spectral.status_code)
        st.error(error_message)
        return None

    df_bg = pd.DataFrame(data_bg).iloc[:1].apply(pd.to_numeric, errors='coerce')
    df_spectral = pd.DataFrame(data_spectral).iloc[:1].apply(pd.to_numeric, errors='coerce')
    wavelength = df_bg.columns

    absorbance = df_bg.div(df_spectral.values).pow(2)
    plt.figure(figsize=(10, 6))
    plt.plot(wavelength, absorbance.iloc[0], marker='o', linestyle='-')
    plt.xlabel('Wavelength')
    plt.ylabel('Absorbance')
    plt.title('Absorbance Spectrum')
    plt.xticks(rotation='vertical')
    st.pyplot(plt)

    absorbance.to_csv('absorbanceData.csv', index=False)
    return absorbance

def load_tf_model(model_dir):
    try:
        if model_dir.endswith('.tflite'):
            interpreter = Interpreter(model_path=model_dir)
            interpreter.allocate_tensors()
            return interpreter
        else:
            return tf.saved_model.load(model_dir)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

def make_prediction_with_tf_model(model, data):
    if isinstance(model, Interpreter):
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        model.set_tensor(input_details[0]['index'], np.array(data, dtype=np.float64).reshape(1, -1))
        model.invoke()
        predictions = model.get_tensor(output_details[0]['index'])
        return predictions
    else:
        pred_function = model.signatures["serving_default"]
        predictions = pred_function(tf.constant([data], dtype=tf.float64))
        output_key = next(iter(predictions.keys()))
        return predictions[output_key].numpy()

def main():
    st.title('Model Prediction App')

    data = retrieve_data()

    if data is not None:
        st.markdown('**Absorbance Data Table:**')
        absorbance_data_df = pd.read_csv('absorbanceData.csv')
        st.dataframe(absorbance_data_df)

        processed_data = data.iloc[0].values  # Assuming data preprocessing is required

        tf_model_folder_path1 = 'reva-lablink-hb-125-(original-data).csv_best_model_2024-02-16_11-47-00_b4_r0.26'
        tf_model_folder_path2 = 'reva-lablink-hb-125-(original-data).csv_r2_0.39_2024-02-15_11-55-27'
        # tf_model_folder_path3 = 'tflite_model.tflite'
        # tf_model_folder_path4 = 'model_new.tflite'

        tf_model1 = load_tf_model(tf_model_folder_path1)
        tf_model2 = load_tf_model(tf_model_folder_path2)
        # tf_model3 = load_tf_model(tf_model_folder_path3)
        # tf_model4 = load_tf_model(tf_model_folder_path4)

        # Make predictions with each TensorFlow model
        predictions1 = make_prediction_with_tf_model(tf_model1, processed_data)
        predictions2 = make_prediction_with_tf_model(tf_model2, processed_data)
        # predictions3 = make_prediction_with_tf_model(tf_model3, processed_data)
        # predictions4 = make_prediction_with_tf_model(tf_model4, processed_data)
        
        # Display predictions for each model
        # for idx, predictions in enumerate([predictions1, predictions2, predictions3, predictions4], start=1):
        #     st.markdown(f'<font size="5"><b>Model {idx} Haemoglobin Value:</b></font>', unsafe_allow_html=True)
        #     predicted_value = predictions[0][0]  # This accesses the first element in the nested structure

        for idx, predictions in enumerate([predictions1, predictions2], start=1):
            st.markdown(f'<font size="5"><b>Model {idx} Haemoglobin Value:</b></font>', unsafe_allow_html=True)
            predicted_value = predictions[0][0]  # This accesses the first element in the nested structure

            # Check if the predicted value is more than 25
            if predicted_value > 25:
                # Display only the word "High" in red for this model
                st.markdown(f'<font size="5"><b>{predicted_value:.1f} g/dL - <span style="color: red;">High</span></b></font>', unsafe_allow_html=True)
            else:
                # If the predicted value does not exceed 25, display the value normally for this model
                st.markdown(f'<font size="5"><b>{predicted_value:.1f} g/dL</b></font>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()

