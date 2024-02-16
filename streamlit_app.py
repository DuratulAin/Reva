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
    # st.write(absorbance)
    # st.write(wavelength)

    absorbance_data = absorbance.iloc[0]
    # st.write(absorbance_data)

    # st.line_chart(absorbance_data)

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

    # Assuming the model expects a specific shape or preprocessing, apply it here
    # For demonstration, using the data directly
    processed_data = data  # Placeholder for any preprocessing needed for your model

    # Load TensorFlow model
    tf_model_folder_path = '/workspaces/Reva/reva-lablink-hb-125-(original-data).csv_r2_0.39_2024-02-15_11-55-27'
    tf_model = load_tf_model(tf_model_folder_path)

    # Optional: Inspect model signature to verify input/output
    # inspect_model_signature(tf_model)

    # Make predictions with the TensorFlow model
    predictions = make_prediction_with_tf_model(tf_model, processed_data)

    # Display predictions - adjust according to your actual model's output
    # st.write("Predictions:", predictions)

    st.markdown('<font size="5"><b>Haemoglobin Value:</b></font>', unsafe_allow_html=True)

    predicted_value = predictions[0][0]  # This accesses the first element in the nested structure
    # Format the prediction with HTML for displaying in Streamlit, ensuring it's displayed as a floating-point number
    st.markdown(f'<font size="5"><b>{predicted_value:.1f} g/dL</b></font>', unsafe_allow_html=True)

    # # Load the CSV data of original data
    # original_data = pd.read_csv('Raw data all.csv')

    # # Combine both datas
    # combined_data = pd.concat([xano_data_df, original_data], ignore_index=True)

    # st.dataframe(xano_data_df)
    # st.dataframe(original_data)  # Uncomment if you want to display the original data as well
    # st.dataframe(combined_data)

    # # Load the UMAP model from the joblib file
    # umap_model = load_model('umap_model_new_10.joblib').transform(combined_data)

    # # Load the Linear Regression model and make a prediction
    # linear_reg_model = load_model('linear_reg_model_new_10.joblib')
    # linear_reg_prediction = linear_reg_model.predict(combined_data)

    # # Load the Decision Tree model and make a prediction
    # decision_tree_model = load_model('decision_tree_model_new_10.joblib')
    # decision_tree_prediction = decision_tree_model.predict(combined_data)

    # # Load the Linear Regression model with UMAP and make prediction
    # linear_reg_model_umap = load_model('linear_reg_model_new_umap_10.joblib')
    # linear_reg_umap_pred = linear_reg_model_umap.predict(umap_model)

    # # Load the Decision Tree model with UMAP and make prediction
    # decision_tree_model_umap = load_model('decision_tree_model_new_umap_10.joblib')
    # decision_tree_umap_pred = decision_tree_model_umap.predict(umap_model)

    # # Display predictions from both models in a larger and bold format
    # st.markdown('<font size="6"><b>Predictions:</b></font>', unsafe_allow_html=True)
    # st.markdown('**Linear Regression Model:**')
    # st.markdown(f'<font size="5"><b>{linear_reg_prediction[0]} g/dL</b></font>', unsafe_allow_html=True)
    # st.markdown('**Decision Tree Model:**')
    # st.markdown(f'<font size="5"><b>{decision_tree_prediction[0]} g/dL</b></font>', unsafe_allow_html=True)
    # st.markdown('**Linear Regression Model with UMAP:**')
    # st.markdown(f'<font size="5"><b>{linear_reg_umap_pred[0]:.1f} g/dL</b></font>', unsafe_allow_html=True)
    # st.markdown('**Decision Tree Model with UMAP:**')
    # st.markdown(f'<font size="5"><b>{decision_tree_umap_pred[0]} g/dL</b></font>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
