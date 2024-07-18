import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load data
file_url = "https://raw.githubusercontent.com/agusheryanto182/FP-big-data/master/data_set/nilai_ujian_sma.csv"
data = pd.read_csv(file_url)

# Convert columns JUMLAH.PESERTA and NILAI to numeric
data['JUMLAH.PESERTA'] = pd.to_numeric(data['JUMLAH.PESERTA'], errors='coerce')
data['NILAI'] = pd.to_numeric(data['NILAI'], errors='coerce')

# Drop rows with NaN values after conversion
data.dropna(subset=['JUMLAH.PESERTA', 'NILAI'], inplace=True)

# Add new features
data['RASIO_PESERTA_NILAI'] = data['JUMLAH.PESERTA'] / data['NILAI']
avg_nilai_per_provinsi = data.groupby('PROVINSI')['NILAI'].transform('mean')
avg_nilai_per_program_studi = data.groupby('PROGRAM.STUDI')['NILAI'].transform('mean')
data['AVG_NILAI_PROVINSI'] = avg_nilai_per_provinsi
data['AVG_NILAI_PROGRAM_STUDI'] = avg_nilai_per_program_studi

# Select columns to use
X_simple = data[['JUMLAH.PESERTA']]
y_simple = data['NILAI']

# Create simple linear regression model
model_simple = LinearRegression()
model_simple.fit(X_simple, y_simple)

# Initialize Streamlit
st.title('Dashboard Education Data Analysis')

# Dropdown for selecting province
provinsi_list = data['PROVINSI'].unique()
selected_provinsi = st.selectbox('Select Province', provinsi_list)

# Dropdown for selecting city
if selected_provinsi:
    kota_list = data[data['PROVINSI'] == selected_provinsi]['KOTA'].unique()
    selected_kota = st.selectbox('Select City', kota_list)

    # Dropdown for selecting school name
    if selected_kota:
        sekolah_list = data[data['KOTA'] == selected_kota]['NAMA.SATUAN.PENDIDIKAN'].unique()
        selected_sekolah = st.selectbox('Select School Name', sekolah_list)

        # Visualization based on user selection
        st.subheader('Visualization Data')

        # Boxplot based on selection
        st.subheader('Boxplot Score Based on Selection')
        if selected_sekolah:
            selected_school_data = data[data['NAMA.SATUAN.PENDIDIKAN'] == selected_sekolah]
            fig, ax = plt.subplots()
            sns.boxplot(x='MATA PELAJARAN', y='NILAI', data=selected_school_data, ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)

        # Heatmap correlation between numerical variables
        st.subheader('Heatmap Correlation Between Numerical Variables')
        sns.set(style='white')
        numerical_cols = ['JUMLAH.PESERTA', 'NILAI', 'RASIO_PESERTA_NILAI', 'AVG_NILAI_PROVINSI', 'AVG_NILAI_PROGRAM_STUDI']
        corr_matrix = data[numerical_cols].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        fig, ax = plt.subplots()
        sns.heatmap(corr_matrix, annot=True, mask=mask, cmap=cmap, ax=ax)
        st.pyplot(fig)
        
        # Polynomial regression and simple linear regression
        st.subheader('Polynomial Regression and Simple Linear Regression')
        if selected_sekolah:
            X_selected = selected_school_data[['JUMLAH.PESERTA']]
            y_selected = selected_school_data['NILAI']
            fig, ax = plt.subplots(figsize=(8, 6))
            plt.scatter(X_selected, y_selected, color='black')
            # Polynomial regression
            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(X_selected)
            model_poly = LinearRegression()
            model_poly.fit(X_poly, y_selected)
            y_pred_poly = model_poly.predict(X_poly)
            plt.plot(X_selected, y_pred_poly, color='blue', linewidth=3, label='Polynomial Regression')
            # Simple linear regression
            model_simple = LinearRegression()
            model_simple.fit(X_selected, y_selected)
            y_pred_simple = model_simple.predict(X_selected)
            plt.plot(X_selected, y_pred_simple, color='red', linewidth=3, label='Simple Linear Regression')
            plt.legend()
            st.pyplot(fig)

            # Polynomial regression
            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(X_selected)
            model_poly = LinearRegression()
            model_poly.fit(X_poly, y_selected)
            y_pred_poly = model_poly.predict(X_poly)

            # Plot polynomial regression
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(X_selected, y_selected, color='black')
            ax.plot(X_selected, y_pred_poly, color='blue', linewidth=3)
            ax.set_title('Polynomial Regression between JUMLAH.PESERTA and NILAI')
            ax.set_xlabel('JUMLAH.PESERTA')
            ax.set_ylabel('NILAI')
            st.pyplot(fig)

            # Simple linear regression
            plt.figure(figsize=(8, 6))
            plt.scatter(X_selected, y_selected, color='black')
            plt.plot(X_selected, model_simple.predict(X_selected), color='blue', linewidth=3)
            plt.title('Simple Linear Regression between JUMLAH.PESERTA and NILAI')
            plt.xlabel('JUMLAH.PESERTA')
            plt.ylabel('NILAI')
            st.pyplot()

# Run Streamlit app
if __name__ == '__main__':
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)

