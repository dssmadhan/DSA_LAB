import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data.csv")
df_encoded = df.copy()

# Encode categorical columns
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Prepare features and target
X = df_encoded.drop(columns=['City_Tier'])
y = df_encoded['City_Tier']

# Setup page
st.set_page_config(page_title="Smart Spending Analyzer", layout="wide", page_icon="💰")

# App header
col1, col2 = st.columns([1, 3])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/477/477103.png", width=100)
with col2:
    st.title("Smart Spending Analyzer")
    st.markdown("Explore, visualize and predict lifestyle behavior based on income and expenses.")

# Sidebar Navigation
with st.sidebar:
    st.header("Navigation")
    section = st.radio("", 
                      ["📊 Univariate Analysis", 
                       "🔗 Bivariate Analysis", 
                       "📉 PCA", 
                       "🔮 Predict City Tier",
                       "🧾 Overview"],
                      index=0)
    
    st.divider()
    st.subheader("Dataset Info")
    st.write(f"Rows: {df.shape[0]}")
    st.write(f"Columns: {df.shape[1]}")
    st.write(f"City Tiers: {df['City_Tier'].nunique()}")

# -------------------------------
# 1. Univariate Analysis
if "Univariate Analysis" in section:
    st.header("Univariate Analysis")
    st.markdown("Explore the distribution of individual variables")
    
    col = st.selectbox("Select a numeric column", 
                      df.select_dtypes(include=['float64', 'int64']).columns,
                      key='uni_col')
    
    tab1, tab2 = st.tabs(["Distribution", "Statistics"])
    
    with tab1:
        fig = px.histogram(df, x=col, marginal="box", nbins=50, 
                          title=f"Distribution of {col}")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.dataframe(df[col].describe().to_frame().style.background_gradient())

# -------------------------------
# 2. Bivariate Analysis
elif "Bivariate Analysis" in section:
    st.header("Bivariate Analysis")
    st.markdown("Explore relationships between two variables")
    
    col1, col2 = st.columns(2)
    with col1:
        x = st.selectbox("X-axis", 
                        df.select_dtypes(include=['float64', 'int64']).columns,
                        key='x_axis')
    with col2:
        y = st.selectbox("Y-axis", 
                        df.select_dtypes(include=['float64', 'int64']).columns, 
                        index=1,
                        key='y_axis')
    
    fig = px.scatter(df, x=x, y=y, color='City_Tier', 
                    title=f"{x} vs {y} by City Tier")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# 3. Multivariate Analysis with PCA
elif "PCA" in section:
    st.header("Principal Component Analysis")
    st.markdown("Visualize high-dimensional data in 2D space")
    
    features = df_encoded.drop(columns=["City_Tier", "Occupation"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    pca_df["City_Tier"] = df_encoded["City_Tier"]
    
    tab1, tab2 = st.tabs(["Visualization", "Details"])
    
    with tab1:
        fig = px.scatter(pca_df, x="PC1", y="PC2", 
                        color=pca_df["City_Tier"].astype(str), 
                        title="PCA: Data in 2D Space")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.info("Explained Variance Ratio")
        st.write(pca.explained_variance_ratio_)
        
        st.info("Component Loadings")
        loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=features.columns)
        st.dataframe(loadings.style.background_gradient(axis=None))

# -------------------------------
# 4. Prediction Interface
elif "Predict City Tier" in section:
    st.header("Predict City Tier")
    st.markdown("Enter your lifestyle and spending details to predict your city tier:")
    
    # Input form with columns
    input_data = {}
    cols = st.columns(2)
    for i, col in enumerate(df.drop(columns=['City_Tier']).columns):
        with cols[i % 2]:
            if df[col].dtype in ['float64', 'int64']:
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                mean_val = float(df[col].mean())
                input_data[col] = st.slider(
                    f"{col}", 
                    min_val, 
                    max_val, 
                    mean_val,
                    help=f"Range: {min_val} to {max_val}"
                )
            else:
                options = df[col].unique().tolist()
                input_data[col] = st.selectbox(
                    f"{col}", 
                    options,
                    help=f"Select from available options"
                )

    # Choose model
    model_name = st.selectbox("Choose Model", 
                              ["Naive Bayes", "Logistic Regression", "SVM", "Linear SVC"])
    
    if model_name == "Naive Bayes":
        model = GaussianNB()
    elif model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "SVM":
        model = SVC(probability=True)
    elif model_name == "Linear SVC":
        model = LinearSVC()

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    model.fit(X_train, y_train)

    if st.button("Predict City Tier"):
        input_df = pd.DataFrame([input_data])
        for col in input_df.select_dtypes(include='object').columns:
            le = label_encoders[col]
            input_df[col] = le.transform(input_df[col])
        
        pred = model.predict(input_df)[0]
        city_result = label_encoders['City_Tier'].inverse_transform([pred])[0]
        st.success(f"Predicted City Tier: {city_result}")
        
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0]
            prob_df = pd.DataFrame({
                'City Tier': label_encoders['City_Tier'].classes_,
                'Probability': proba
            })
            fig = px.bar(prob_df, x='City Tier', y='Probability', 
                         color='Probability',
                         title="Prediction Confidence")
            st.plotly_chart(fig, use_container_width=True)

    # Evaluation
    st.divider()
    if st.checkbox("Show Model Evaluation Metrics"):
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        st.subheader("Classification Report")
        rep_df = pd.DataFrame(report).transpose()
        st.dataframe(rep_df.style.background_gradient(axis=0, cmap='YlGnBu'))

        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

# -------------------------------
# 5. Dataset Overview
elif "Overview" in section:
    st.header("Indian Personal Finance Dataset Overview")
    st.markdown("""
    This dataset captures financial behavior of Indian consumers across income levels, occupations, and city tiers.
    
    **Features include:**
    - Monthly Income
    - Spending on essentials (food, rent, etc.)
    - Non-essential expenses (entertainment, travel)
    - Savings and loans
    - City Tier classification (1, 2, 3)
    
    **Applications:**
    - Regional segmentation
    - Financial behavior analysis
    - Credit scoring and marketing strategy
    """)

    st.subheader("First 10 Rows")
    st.dataframe(df.head(10))

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

# Footer
st.divider()
st.caption("Smart Spending Analyzer • Built with Streamlit")
