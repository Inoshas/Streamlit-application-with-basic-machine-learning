import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from sklearn.preprocessing import PolynomialFeatures



########################
# Combine data to one file
#df_consumption = pd.read_excel('data/consumptions.xlsx')

########################
# Load Data
#df = pd.read_csv("data/fmi_weather_and_price.csv", parse_dates=['Time'])


# Streamlit App
st.title("Finland total energy production and weather impact")
st.write("""In this analysis we are trying to investigate the correlation between weather and the electricity production in Finland.
         The data was obtained by the Fingrid official website from 2022 March to 2023 November.
         Then weather data was obtained from the shared data file during the data visualization course.
         This has 3 subsections: such as regression approcah, neural network approach and the comparison between the results. """)

st.subheader("Regression Approach")

st.write(" ##### Data Cleaning:")
st.write("""
Fingrid's total power generation data includes multiple sources: **nuclear, hydro, solar, wind, small-scale, and other resources**.
However, in this analysis, we only consider four sources: **wind, solar, hydro, and nuclear**.
As a result, the **Total** production should always be **greater than or equal to** the sum of these four sources.
Any entries where **Total** is less than the sum of Wind, Solar, Hydro, and Nuclear are considered inconsistent and are removed from the dataset. The table below lists the removed entries.
""")
df = pd.read_csv("data/filtered_data.csv", parse_dates=['Time'])
df_clean = df[['Solar','Hydro','Wind_Power','Nuclear','Total' ,'Wind_Weather','Temp']].dropna()
df_invalid_total = df_clean[df_clean['Total'] < (df_clean['Solar'] + df_clean['Wind_Power'] + df_clean['Hydro'] + df_clean['Nuclear'])]

st.write(df_invalid_total)

st.write(" ##### Linear regression:")
#st.write(df.columns)
#'Time' #'total'
df_clean = df_clean[df_clean['Total'] >= (df_clean['Solar'] + df_clean['Wind_Power'] + df_clean['Hydro'] + df_clean['Nuclear'])]
X = df_clean[[ 'Hydro','Solar','Wind_Power','Nuclear','Wind_Weather','Temp']]
Y = df_clean[['Total' ]]

#X['Time'] = X['Time'].astype('int64') // 10**9  # Convert datetime to Unix timestamp
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
Y_scaled = scaler.fit_transform(Y)
# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=0)
model = LinearRegression()
model.fit(X_train, Y_train)


#get predictions
Y_pred = model.predict(X_test)

mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)

st.write(f"RMSE: {rmse:.3f}")  # Round to 3 decimal places
st.write(f"R² value: {model.score(X_test, Y_test):.3f}")  # Round to 3 decimal places

st.write(f""" Linear regression model achieved an RMSE of {rmse:.3f} and an R² value of {model.score(X_test, Y_test):.3f},
         indicating a strong correlation between the selected features and total power generation. 
         The R² value of {model.score(X_test, Y_test):.3f} suggests that approximately {model.score(X_test, Y_test):.1f}% of the variation in total power generation is explained by the model, demonstrating a good fit.""")


# Create scatter plot
fig = px.scatter(
    x=Y_test.flatten(), 
    y=Y_pred.flatten(),  
    labels={'x': 'True', 'y': 'Predicted'}
)

fig.update_layout(title='True vs Predicted values')

# Add a trendline manually (red)
m, b = np.polyfit(Y_test.flatten(), Y_pred.flatten(), 1)  # Linear regression fit
x_range = np.linspace(min(Y_test.flatten()), max(Y_test.flatten()), 100)  # Generate x values
y_range = m * x_range + b  # Compute y values

fig.update_traces(marker=dict(color='blue', size=5)) 

fig.add_trace(go.Scatter(
    x=x_range, 
    y=y_range, 
    mode='lines', 
    line=dict(color='red', width=3), 
    name="Trendline",
    showlegend=True
))

st.plotly_chart(fig)



###############################################
# Assuming `model` is already trained
feature_names = X.columns
coefficients = np.abs(model.coef_).flatten()  # Ensure coefficients are 1D

# Ensure coefficients are correctly converted into a NumPy array
coefficients = np.array(coefficients, dtype=float)

# Create DataFrame for feature importance
feature_importance = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

# Ensure sorting works correctly
feature_importance = feature_importance.sort_values(by='Coefficient', ascending=False)

# Display in Streamlit
st.write(" ###### Feature Importance Analysis")
st.dataframe(feature_importance)  # Display as interactive table


st.markdown("""
- **Hydro (0.4663)** and **Temperature (0.4610)** have the highest impact on total power generation, meaning fluctuations in these variables significantly influence the model’s predictions.
- **Wind Power (0.3745)** and **Nuclear (0.2912)** also play a substantial role, but their impact is slightly lower than hydro and temperature.
- **Solar (0.0315)** has a very small effect, which is expected due to Finland's limited solar generation potential.
- **Wind Weather (0.0033)** has the lowest impact, indicating that wind speed alone may not be a strong predictor of total generation, likely due to other influencing factors like turbine efficiency or operational constraints.
""")

st.write(" ##### Polynomial regression:")
st.write( "First we are going to analyze the perfromance levels for different polynomials regressions. Below table illustares the RMSE value and R² value for different polynomial values.")



# Define the polynomial degrees to test
degrees = [2, 3, 4, 5, 6,7]

# Lists to store results
degree_list = []
rmse_list = []
r2_list = []

for degree in degrees:
    # Transform features using polynomial expansion
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)  # Use transform() instead of fit_transform()

    # Train model
    model_poly = LinearRegression()
    model_poly.fit(X_train_poly, Y_train)

    # Get predictions
    Y_pred_poly = model_poly.predict(X_test_poly)

    # Calculate RMSE and R²
    rmse_poly = np.sqrt(mean_squared_error(Y_test, Y_pred_poly))
    r2_poly = model_poly.score(X_test_poly, Y_test)

    # Store results in lists
    degree_list.append(degree)
    rmse_list.append(rmse_poly)
    r2_list.append(r2_poly)

# Create DataFrame for results
results_df = pd.DataFrame({
    'Polynomial Degree': degree_list,
    'RMSE': rmse_list,
    'R² Score': r2_list
})

st.write(results_df)

st.markdown("""
- As the polynomial degree increases from **2 to 6**, **RMSE decreases** and **R² increases** , showing improved accuracy.
- **Degree 5** and **Degree 6** provide the best balance of accuracy without overfitting.
- **Degree 7** shows a **higher RMSE and lower R²**, suggesting overfitting.
- **Conclusion:** **Degree 5 or 6** is optimal, capturing complex patterns while maintaining generalizability.
""")
st.write( "Below graph illustrates the degree 5 polynomial outcome")

poly = PolynomialFeatures(degree=5)
X_train_poly = poly.fit_transform(X_train)

model_poly = LinearRegression()
model_poly.fit(X_train_poly, Y_train)

X_test_poly = poly.fit_transform(X_test)
Y_pred_poly = model_poly.predict(X_test_poly)




fig1 = px.scatter(
    x=Y_test.flatten(),  
    y=Y_pred_poly.flatten(),  
    labels={'x': 'True', 'y': 'Predicted'}, 
    title='True vs Predicted values', 
    trendline='ols'  # Automatically adds a trendline
)



fig1.update_traces(line=dict(color='red'))
fig1.update_traces(marker=dict(color='blue', size=6))  # Change scatter point color



st.plotly_chart(fig1)


X_train_poly = poly.fit_transform(X_train)
feature_names = poly.get_feature_names_out(X.columns)

# Convert coefficients to 1D array (if necessary)
coefficients = np.abs(model_poly.coef_.ravel())  # Ensure it's 1D

# Create DataFrame for feature importance
feature_importance = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

# Sort by importance
feature_importance = feature_importance.sort_values(by='Coefficient', ascending=False)

# Display in Streamlit
st.write(" **Feature Importance in Polynomial Regression (Degree 5)**")
st.dataframe(feature_importance)  # Display as interactive table
# Display in Streamlit


st.write("""
Nuclear power plays a dominant role with strong nonlinear effects in 
nuclear power’s impact on total generation. Hydro power, and temperature interacts 
with multiple features remain significants too. Also this highlights that relationships 
between features matter. """)
 

from sklearn.ensemble import RandomForestRegressor
st.write(" ##### Random forest regression:")

# Train Random Forest on the original features (not polynomial-expanded ones)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, Y_train.ravel())  # Flatten Y_train

# Get feature importance
feature_importance_rf = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Display in Streamlit
st.write(" **Feature Importance from Random Forest**")
st.dataframe(feature_importance_rf)

# Visualize feature importance using Plotly
fig = px.bar(
    feature_importance_rf,
    x='Feature',
    y='Importance',
    title='Feature Importance from Random Forest',
    labels={'Importance': 'Feature Importance'},
)

st.plotly_chart(fig)

st.write("## Summary: ")

st.write("In the regression model, nuclear power emerges as the most significant contributor to total production. This aligns with the broader energy landscape, where renewable sources still play a relatively smaller role in Finland’s electricity generation. However, the neural network analysis reveals a different perspective, uncovering deeper relationships between renewable energy sources and production. More details on these insights can be found on the next page.")

###################################

