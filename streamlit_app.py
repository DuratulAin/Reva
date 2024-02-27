import streamlit as st
import pandas as pd
import joblib
import requests
import matplotlib.pyplot as plt
import tensorflow as tf

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

    # Extract first line and convert to numeric
    df_bg = pd.DataFrame(data_bg).iloc[:1].apply(pd.to_numeric, errors='coerce')
    df_spectral = pd.DataFrame(data_spectral).iloc[:1].apply(pd.to_numeric, errors='coerce')
    wavelength = df_bg.columns

    # Calculate absorbance
    absorbance = df_bg.div(df_spectral.values).pow(2)

    absorbance_data = absorbance.iloc[0]

    # Plotting the line graph of absorbance data with markers
    plt.figure(figsize=(10, 6))
    plt.plot(wavelength, absorbance.iloc[0], marker='o', linestyle='-')  # Add marker here
    plt.xlabel('Wavelength')
    plt.ylabel('Absorbance')
    plt.title('Absorbance Spectrum')
    plt.xticks(rotation='vertical')
    st.pyplot(plt)

    absorbance.to_csv('absorbanceData.csv', index=False)
    return absorbance

# Function to load a model from a pickle file
def load_model(model_file):
    with open(model_file, 'rb') as f:
        model = joblib.load(f)
    return model

# Main Streamlit app
def main():
    st.title('Model Prediction App')

    # Retrieve and plot data
    retrieve_data()  # Ensures the absorbanceData.csv is created

    # Load the CSV data from Xano
    xano_data_df = pd.read_csv('absorbanceData.csv')

# New function to load a TensorFlow model from a directory
def load_tf_model(model_dir):
    return tf.saved_model.load(model_dir)

# Function to inspect model signature
def inspect_model_signature(model):
    for key, value in model.signatures.items():
        print('Signature:', key)
        print('Inputs:', value.inputs)
        for output_key, output_value in value.outputs.items():
            print('Output key:', output_key)
        print('---')

# Function to make predictions with the loaded TensorFlow model
def make_prediction_with_tf_model(model, data):
    pred_function = model.signatures["serving_default"]
    predictions = pred_function(tf.constant(data, dtype=tf.float64))
    # Inspect your model signature to find the correct output key
    # Adjust 'output_0' or any other key according to your model's signature
    output_key = next(iter(predictions.keys()))  # Using the first output key
    return predictions[output_key].numpy()

# Main Streamlit app
def main():
    st.title('Model Prediction App')

    # Retrieve and plot data
    data = retrieve_data()

    # Display the absorbance data table
    st.markdown('**Absorbance Data Table:**')
    absorbance_data_df = pd.read_csv('absorbanceData.csv')
    st.dataframe(absorbance_data_df)

    # Assuming the model expects a specific shape or preprocessing, apply it here
    # For demonstration, using the data directly
    processed_data = data  # Placeholder for any preprocessing needed for your model

    # Assuming you have two more model folder paths
    tf_model_folder_path1 = 'reva-lablink-hb-125-(original-data).csv_r2_0.39_2024-02-15_11-55-27'
    tf_model_folder_path2 = 'tflite_model.tflite'
    tf_model_folder_path3 = 'model_new.tflite'
    
    # Load TensorFlow models
    tf_model1 = load_tf_model(tf_model_folder_path1)
    tf_model2 = load_tf_model(tf_model_folder_path2)
    tf_model3 = load_tf_model(tf_model_folder_path3)
    
    # Optional: Inspect model signature to verify input/output for each model
    # inspect_model_signature(tf_model1)
    # inspect_model_signature(tf_model2)
    # inspect_model_signature(tf_model3)
    
    # Make predictions with each TensorFlow model
    predictions1 = make_prediction_with_tf_model(tf_model1, processed_data)
    predictions2 = make_prediction_with_tf_model(tf_model2, processed_data)
    predictions3 = make_prediction_with_tf_model(tf_model3, processed_data)
    
    # Display predictions for each model
    for idx, predictions in enumerate([predictions1, predictions2, predictions3], start=1):
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
