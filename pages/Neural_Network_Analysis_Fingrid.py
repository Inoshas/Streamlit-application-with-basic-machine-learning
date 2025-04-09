import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

from sklearn.inspection import permutation_importance


from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import io
from contextlib import redirect_stdout

st.subheader("Neural Network Approach")

st.write("""
In this work, we are going to analyze the data using a machine learning approach. 
We consider a neural network approach, and suitable hyperparameters were found 
after testing with different parameter values. The testing has been done for different hyperparamerets
and below illustrates the best outcome.
""")

# 🚀 Cache data loading for performance
@st.cache_data
def load_data():
    df = pd.read_csv("data/filtered_data.csv", parse_dates=['Time'])
    df_clean = df[['Solar', 'Hydro', 'Wind_Power', 'Nuclear', 'Total', 'Wind_Weather', 'Temp']].dropna()
    df_clean = df_clean[df_clean['Total'] >= (df_clean['Solar'] + df_clean['Wind_Power'] + df_clean['Hydro'] + df_clean['Nuclear'])]
    return df_clean

df_clean = load_data()

# Define features and labels
X = df_clean[['Hydro', 'Solar', 'Wind_Power', 'Nuclear', 'Wind_Weather', 'Temp']].values.astype(np.float32)
Y = df_clean[['Total']].values.astype(np.float32)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 🚀 Use MinMaxScaler for better stability
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train = scaler_x.fit_transform(x_train)
X_test = scaler_x.transform(x_test)

y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

# 🎯 Neural Network Architecture Optimized
model = keras.Sequential([
    keras.Input(shape=(X_train.shape[1],)),  # Input layer
    Dense(128, activation='relu'),
    Dropout(0.2),  # Dropout for regularization
    Dense(64, activation='elu'),
    Dropout(0.1),
    Dense(1, activation='elu')  # ELU is better for regression tasks
])

# ✅ Use Adam with a lower learning rate for better convergence
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='mse',
              metrics=[tf.keras.metrics.RootMeanSquaredError()])

# ✅ Early stopping to prevent overfitting
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# ✅ Capture model summary output safely
buffer = io.StringIO()
with redirect_stdout(buffer):
    model.summary()
st.text(buffer.getvalue())

# ✅ Custom Streamlit Callback
class StreamlitCallback(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.epoch_progress = st.progress(0)
        self.epoch_text = st.empty()
        self.loss_chart = st.line_chart([])

    def on_epoch_end(self, epoch, logs=None):
        epochs = self.params.get('epochs', 100)  # Default to 100 if unavailable
        self.epoch_progress.progress((epoch + 1) / epochs)
        self.epoch_text.text(f"Epoch {epoch + 1}/{epochs} - Loss: {logs['loss']:.4f}")
        #self.loss_chart.line_chart(pd.DataFrame({"Training Loss": [logs["loss"]], "Validation Loss": [logs["val_loss"]]}))

# ✅ Train Model (Only Once)
history = model.fit(X_train, y_train,
                    epochs=100,
                    batch_size=256,  # Increased batch size for performance
                    validation_data=(X_test, y_test),
                    callbacks=[early_stop, StreamlitCallback()])

# ✅ Extract loss and validation loss from training history
loss_df = pd.DataFrame({
    'Epoch': range(1, len(history.history['loss']) + 1),
    'Training Loss': history.history['loss'],
    'Validation Loss': history.history['val_loss']
})

# ✅ Plot loss curves in Streamlit
st.subheader("Loss Curve")
fig_loss = px.line(loss_df, x='Epoch', y=['Training Loss', 'Validation Loss'], 
                   labels={'value': 'Loss', 'variable': 'Loss Type'},
                   title="Training & Validation Loss Over Epochs")
st.plotly_chart(fig_loss)

# ✅ Make predictions
predictions = model.predict(X_test)

# ✅ Compute R² Score (Rescale y back to original scale)
y_pred_rescaled = scaler_y.inverse_transform(predictions)
y_test_rescaled = scaler_y.inverse_transform(y_test)

r2 = r2_score(y_test_rescaled, y_pred_rescaled)
st.write(f" **R² Score:** {r2:.4f} (Closer to 1 is better)")

# ✅ Compute Feature Importance using Permutation Importance
perm_importance = permutation_importance(model, X_test, y_test, scoring='r2')

# Convert to DataFrame
feature_importance_df = pd.DataFrame({
    'Feature': ['Hydro', 'Solar', 'Wind_Power', 'Nuclear', 'Wind_Weather', 'Temp'],
    'Importance': perm_importance.importances_mean
}).sort_values(by="Importance", ascending=False)

# ✅ Display in Streamlit
st.subheader("Feature Importance in Neural Network")
st.write("The following chart shows how much each feature contributes to the total energy production prediction.")

# Plot Feature Importance
fig_importance = px.bar(feature_importance_df, x='Importance', y='Feature', 
                         orientation='h', title="Feature Importance (Permutation Importance)",
                         labels={'Importance': 'Mean Decrease in R² Score'})
st.plotly_chart(fig_importance)


# ✅ Scatter Plot: True vs. Predicted Values
fig = px.scatter(
    x=y_test_rescaled.flatten(), 
    y=y_pred_rescaled.flatten(),
    labels={'x': 'True Values', 'y': 'Predicted Values'},
    title="Neural Network Predictions vs. Actual",
    opacity=0.7  # Make points slightly transparent
)

# ✅ Add Perfect Prediction Line (Diagonal Reference Line)
min_val = min(y_test_rescaled.min(), y_pred_rescaled.min())
max_val = max(y_test_rescaled.max(), y_pred_rescaled.max())

fig.add_trace(go.Scatter(
    x=[min_val, max_val], 
    y=[min_val, max_val], 
    mode='lines', 
    line=dict(color='red', width=5), 
    name="Perfect Prediction Line")
)

# ✅ Show the plot in Streamlit
st.plotly_chart(fig)

st.write("""
📌 **Observations:**
- The closer the points are to the red line, the better the model performance.
- Increasing the dataset size and fine-tuning hyperparameters can improve accuracy.
- Experiment with different activation functions and hidden layers.
""")
# Compute correlations with temperature
correlations = df_clean[['Total', 'Wind_Power', 'Solar', 'Hydro', 'Nuclear']].corrwith(df_clean['Temp'])

# Convert to DataFrame for display
correlation_table = pd.DataFrame({'Energy Source': correlations.index, 'Correlation with Temperature': correlations.values})

# Round for better readability
correlation_table['Correlation with Temperature'] = correlation_table['Correlation with Temperature'].round(3)

# Display in Streamlit
st.write("### Correlation of Temperature with Different Energy Sources")
st.write("""Since temperature is impacting overall, the correlation between different energy productions and temperature has been calculated below. Though there is no direct impact on nuclear production
         from temperature, nuclear balance the demand which highly depend on the weather and time of the day.""")
st.dataframe(correlation_table)


import plotly.express as px

# Create a bar chart
fig = px.bar(
    correlation_table,
    x='Energy Source',
    y='Correlation with Temperature',
    text='Correlation with Temperature',
    labels={'Correlation with Temperature': 'Correlation Coefficient'},
    title="Correlation of Temperature with Different Energy Sources",
    color='Correlation with Temperature',
    color_continuous_scale='RdBu'  # Red for negative, blue for positive
)

# Update layout for better readability
fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
fig.update_layout(yaxis=dict(title="Correlation Coefficient", range=[-1, 1]))

# Display in Streamlit
st.plotly_chart(fig)


import streamlit as st

st.subheader("Impact of Temperature on Energy Production: Regression vs Neural Networks")

st.markdown("""
The analysis highlights two key perspectives on how temperature influences Finland's electricity production:

- **Regression Analysis (including Random Forest)** identifies **nuclear power** as the most significant factor influencing total electricity production. This suggests that nuclear plays a stabilizing role in the energy mix, with variations in other sources being compensated by nuclear generation.
- **Neural Network Analysis**, on the other hand, captures the broader relationships among features and shows that **temperature has a notable impact on all renewable energy sources** (wind, solar, and hydro). This indicates that neural networks can better detect the **interconnected dependencies** between temperature and various production sources, rather than isolating a single dominant factor like nuclear **. Further, hydro is also impact on total production.

### Conclusion
While regression models highlight the **direct impact of nuclear power on total production**, neural networks reveal the **overall influence of temperature** on the renewable energy mix. This suggests that temperature-driven fluctuations in **wind, solar, and hydro generation** indirectly affect total production, requiring adjustments from nuclear and other sources.
""")
