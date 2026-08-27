# App import
import streamlit as st

# Pandas
import pandas as pd

# Libraries for model training
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, classification_report, r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

# Libraries for visualization
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio

st.set_page_config(page_title="DataPilot | Train a Model", layout="wide")

st.markdown("""
<style>
/* ── ROOT TOKENS (match DataPilot) ─────────────── */
:root {
    --primary:      #2563EB;
    --primary-dark: #1d4ed8;
    --secondary:    #059669;
    --bg-warm:      #F5F4EF;
    --card-bg:      #FFFFFF;
    --text-main:    #111827;
    --text-muted:   #6B7280;
    --border:       #E5E7EB;
    --radius:       12px;
    --shadow:       0 1px 3px rgba(0,0,0,0.08);
    --shadow-hover: 0 4px 12px rgba(0,0,0,0.10);
}

/* ── PAGE BACKGROUND ────────────────────────────── */
.stApp {
    background-color: #F5F4EF !important;
}

/* ── HIDE DEFAULT STREAMLIT CHROME ──────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 2.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1100px !important;
}

/* ── HEADINGS & BODY TEXT ───────────────────────── */
h1, h2, h3, h4, h5, h6 {
    color: #111827 !important;
    letter-spacing: -0.02em;
}
.stSubheader > div {
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    color: #111827 !important;
    padding-bottom: 0.35rem;
    border-bottom: 2px solid #E5E7EB;
    margin-bottom: 0.75rem;
}
[data-testid="stWidgetLabel"] p {
    color: #111827 !important;
}

/* ── CARDS — wrap sections ──────────────────────── */
[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.07) !important;
    margin-bottom: 0.75rem !important;
}
[data-testid="stExpander"]:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.09) !important;
}
[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    color: #111827 !important;
    padding: 0.9rem 1.1rem !important;
}

/* ── BUTTONS ────────────────────────────────────── */
.stButton > button {
    background-color: #2563EB !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    padding: 0.55rem 1.25rem !important;
    transition: background 0.2s ease, box-shadow 0.2s ease !important;
    box-shadow: 0 1px 3px rgba(37,99,235,0.25) !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    background-color: #1d4ed8 !important;
    box-shadow: 0 4px 12px rgba(37,99,235,0.30) !important;
}
.stButton > button:active {
    background-color: #1e40af !important;
    transform: translateY(1px) !important;
}

/* ── DOWNLOAD BUTTON ────────────────────────────── */
[data-testid="stDownloadButton"] > button {
    background-color: #F3F4F6 !important;
    color: #111827 !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background-color: #E5E7EB !important;
}

/* ── FORM SUBMIT BUTTON ─────────────────────────── */
[data-testid="stFormSubmitButton"] > button {
    background-color: #2563EB !important;
    color: white !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    border: none !important;
}

/* ── FILE UPLOADER ──────────────────────────────── */
[data-testid="stFileUploader"] {
    background: #FFFFFF !important;
    border: 2px dashed #D1D5DB !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #2563EB !important;
}

/* ── DATAFRAME / TABLE ──────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid #E5E7EB !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
}
[data-testid="stDataFrame"] table {
    border-collapse: collapse !important;
}
[data-testid="stDataFrame"] thead th {
    background-color: #F9FAFB !important;
    color: #374151 !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    border-bottom: 1px solid #E5E7EB !important;
    padding: 0.65rem 1rem !important;
}
[data-testid="stDataFrame"] tbody td {
    font-size: 0.875rem !important;
    color: #111827 !important;
    padding: 0.55rem 1rem !important;
    border-bottom: 1px solid #F3F4F6 !important;
}
[data-testid="stDataFrame"] tbody tr:hover td {
    background-color: #EFF6FF !important;
}

/* ── METRICS ────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 12px !important;
    padding: 1rem 1.25rem !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    color: #6B7280 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: #2563EB !important;
}

/* ── SELECTBOX & INPUTS ─────────────────────────── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    background: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 8px !important;
    font-size: 0.875rem !important;
    color: #111827 !important;
    transition: border-color 0.2s;
}
[data-testid="stSelectbox"] > div > div:focus-within,
[data-testid="stMultiSelect"] > div > div:focus-within {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.12) !important;
}
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    background: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 8px !important;
    font-size: 0.875rem !important;
    color: #111827 !important;
    padding: 0.5rem 0.75rem !important;
}
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextInput"] input:focus {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.12) !important;
    outline: none !important;
}

/* ── RADIO BUTTONS ──────────────────────────────── */
[data-testid="stRadio"] label {
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    color: #374151 !important;
}
[data-testid="stRadio"] [data-checked="true"] {
    color: #2563EB !important;
}

/* ── SLIDER ─────────────────────────────────────── */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: #2563EB !important;
    border-color: #2563EB !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] [data-testid="stThumbValue"] {
    color: #2563EB !important;
    font-weight: 600 !important;
}

/* ── ALERTS ─────────────────────────────────────── */
[data-testid="stAlert"][data-baseweb="notification"] {
    border-radius: 10px !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
}
.stSuccess {
    background-color: #ECFDF5 !important;
    border-left: 4px solid #059669 !important;
    color: #065F46 !important;
    border-radius: 10px !important;
}
.stWarning {
    background-color: #FFFBEB !important;
    border-left: 4px solid #F59E0B !important;
    color: #92400E !important;
    border-radius: 10px !important;
}
.stError {
    background-color: #FEF2F2 !important;
    border-left: 4px solid #EF4444 !important;
    color: #991B1B !important;
    border-radius: 10px !important;
}
.stInfo {
    background-color: #EFF6FF !important;
    border-left: 4px solid #2563EB !important;
    color: #1E40AF !important;
    border-radius: 10px !important;
}

/* ── STATUS BOX ─────────────────────────────────── */
[data-testid="stStatus"] {
    background: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 12px !important;
    font-size: 0.875rem !important;
}

/* ── CODE BLOCK ─────────────────────────────────── */
[data-testid="stCode"] {
    border-radius: 10px !important;
    border: 1px solid #E5E7EB !important;
    font-size: 0.82rem !important;
}

/* ── DIVIDER ────────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid #E5E7EB !important;
    margin: 1.75rem 0 !important;
}

/* ── CAPTION / HELPER TEXT ──────────────────────── */
[data-testid="stCaptionContainer"] {
    color: #6B7280 !important;
    font-size: 0.8rem !important;
}

/* ── PLOTLY CHART CONTAINER ─────────────────────── */
[data-testid="stPlotlyChart"] {
    background: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 12px !important;
    padding: 0.5rem !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
}

/* ── MULTISELECT TAGS ───────────────────────────── */
[data-baseweb="tag"] {
    background-color: #EFF6FF !important;
    color: #2563EB !important;
    border-radius: 6px !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
}
</style>

<!-- DataPilot Navbar -->
""", unsafe_allow_html=True)

# ── SESSION STATE INIT ──────────────────────────
if "trained_model" not in st.session_state:
    st.session_state.trained_model = None
if "training_meta" not in st.session_state:
    st.session_state.training_meta = None
if "all_trained_models" not in st.session_state:
    st.session_state.all_trained_models = []   # list of {label, model, meta}
if "plotted_graphs" not in st.session_state:
    st.session_state.plotted_graphs = []       # list of {label, fig}
if "training_mode" not in st.session_state:
    st.session_state.training_mode = True      # show training form by default
if "confirmed_clear" not in st.session_state:
    st.session_state.confirmed_clear = False
if "last_file_name" not in st.session_state:
    st.session_state.last_file_name = None
if "pending_clear" not in st.session_state:
    st.session_state.pending_clear = False

st.html("<h1 align='center'> Train any Model </h1>")
file = st.file_uploader(label="Upload a data file", type=[".csv", ".xlsx"], accept_multiple_files=False)

# ── FILE REMOVAL DETECTION ──────────────────────
# If there was a file before and now there isn't — file was removed
if st.session_state.last_file_name is not None and file is None:
    st.session_state.pending_clear = True

# ── CONFIRMATION DIALOG ─────────────────────────
if st.session_state.pending_clear:
    st.warning(
        "⚠️ You removed the uploaded file. This will clear all cleaned data, "
        "model results, and session state. **This cannot be undone.** "
        "Do you want to proceed?"
    )
    col_yes, col_no = st.columns(2)
    with col_yes:
        if st.button("✅ Yes, clear everything", use_container_width=True):
            # Wipe all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("Session cleared. Please upload a new file.")
            st.stop()
    with col_no:
        if st.button("❌ No, keep my data", use_container_width=True):
            st.session_state.pending_clear = False
            st.info("Data preserved. Please re-upload the same file to continue.")
            st.stop()
    st.stop()  # Block rest of app from rendering while dialog is active

# ── TRACK CURRENT FILE NAME ─────────────────────
if file is not None:
    st.session_state.last_file_name = file.name
    st.session_state.pending_clear = False

if file is not None:
    file.seek(0)
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    elif file.name.endswith(".xlsx"):
        df = pd.read_excel(file)
    
    st.dataframe(data=df)

    # ─────────────────────────────────────────────
    # CLEAN DATA SECTION
    # ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🧹 Clean Data before Training (Recommended)")

    # Work on a copy so original df is preserved
    if "cleaned_df" not in st.session_state:
        st.session_state.cleaned_df = df.copy()

    cleaned_df = st.session_state.cleaned_df

    # ── 1. NULL VALUES ──────────────────────────
    st.markdown("#### 1. Null Values")
    null_counts = cleaned_df.isnull().sum()
    null_cols = null_counts[null_counts > 0]

    if null_cols.empty:
        st.success("No null values found.")
    else:
        null_df = pd.DataFrame({
            "Column": null_cols.index,
            "Null Count": null_cols.values
        })
        st.dataframe(null_df, use_container_width=True)

        null_action = st.radio(
            "How do you want to handle null values?",
            ["Drop rows with nulls", "Fill with Mean", "Fill with Median", "Fill with Custom Value"],
            horizontal=True
        )

        if null_action == "Fill with Custom Value":
            custom_val = st.text_input("Enter custom fill value (applied to all null columns):")

        if st.button("Apply Null Fix"):
            if null_action == "Drop rows with nulls":
                cleaned_df = cleaned_df.dropna()
                st.success("Dropped all rows with null values.")
            elif null_action == "Fill with Mean":
                numeric_cols = cleaned_df.select_dtypes(include="number").columns
                cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].mean())
                st.success("Filled null values with column mean.")
            elif null_action == "Fill with Median":
                numeric_cols = cleaned_df.select_dtypes(include="number").columns
                cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].median())
                st.success("Filled null values with column median.")
            elif null_action == "Fill with Custom Value":
                if custom_val:
                    cleaned_df = cleaned_df.fillna(custom_val)
                    st.success(f"Filled null values with '{custom_val}'.")
                else:
                    st.warning("Please enter a custom value.")
            st.session_state.cleaned_df = cleaned_df

    # ── 2. DUPLICATES ───────────────────────────
    st.markdown("#### 2. Duplicate Rows")
    dup_count = cleaned_df.duplicated().sum()

    if dup_count == 0:
        st.success("No duplicate rows found.")
    else:
        st.warning(f"{dup_count} duplicate row(s) found.")
        if st.button("Remove Duplicates"):
            cleaned_df = cleaned_df.drop_duplicates()
            st.session_state.cleaned_df = cleaned_df
            st.success(f"Removed {dup_count} duplicate row(s).")

    # ── 3. OUTLIERS (IQR) ───────────────────────
    st.markdown("#### 3. Outliers (IQR Method)")
    numeric_cols = cleaned_df.select_dtypes(include="number").columns.tolist()

    if not numeric_cols:
        st.info("No numeric columns available for outlier detection.")
    else:
        outlier_cols = st.multiselect(
            "Select columns to check for outliers:",
            options=numeric_cols,
            default=numeric_cols
        )

        if outlier_cols:
            # Calculate outlier counts per column
            outlier_report = {}
            for col in outlier_cols:
                Q1 = cleaned_df[col].quantile(0.25)
                Q3 = cleaned_df[col].quantile(0.75)
                IQR = Q3 - Q1
                outlier_mask = (cleaned_df[col] < Q1 - 1.5 * IQR) | (cleaned_df[col] > Q3 + 1.5 * IQR)
                count = outlier_mask.sum()
                if count > 0:
                    outlier_report[col] = count

            if not outlier_report:
                st.success("No outliers detected in selected columns.")
            else:
                outlier_df = pd.DataFrame({
                    "Column": list(outlier_report.keys()),
                    "Outlier Count": list(outlier_report.values())
                })
                st.dataframe(outlier_df, use_container_width=True)

                if st.button("Remove Outliers"):
                    mask = pd.Series([True] * len(cleaned_df), index=cleaned_df.index)
                    for col in outlier_cols:
                        Q1 = cleaned_df[col].quantile(0.25)
                        Q3 = cleaned_df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        mask &= ~((cleaned_df[col] < Q1 - 1.5 * IQR) | (cleaned_df[col] > Q3 + 1.5 * IQR))
                    before = len(cleaned_df)
                    cleaned_df = cleaned_df[mask]
                    st.session_state.cleaned_df = cleaned_df
                    st.success(f"Removed {before - len(cleaned_df)} outlier row(s).")

    # ── CLEANED DATA PREVIEW ────────────────────
    st.markdown("#### ✅ Cleaned Data Preview")
    st.caption(f"Shape: {cleaned_df.shape[0]} rows × {cleaned_df.shape[1]} columns")
    st.dataframe(st.session_state.cleaned_df, use_container_width=True)

# ─────────────────────────────────────────────
    # MODEL TRAINING SECTION
    # ─────────────────────────────────────────────
    if st.session_state.training_mode:
        st.markdown("---")
        st.subheader("🤖 Train a Model")

        cleaned_df = st.session_state.cleaned_df
        all_columns = cleaned_df.columns.tolist()
        numeric_cols = cleaned_df.select_dtypes(include="number").columns.tolist()

        # ── TASK TYPE ───────────────────────────────
        st.markdown("#### 1. Select Task Type")
        task_type = st.radio("What kind of model do you want to train?",
                            ["Regression", "Classification"], horizontal=True)

        # ── MODEL SELECTION ─────────────────────────
        st.markdown("#### 2. Select Model")
        if task_type == "Regression":
            model_choice = st.selectbox("Choose a Regression model:", [
                "Linear Regression",
                "Multiple Linear Regression",
                "Polynomial Regression"
            ])
        else:
            model_choice = st.selectbox("Choose a Classification model:", [
                "Decision Tree",
                "K-Nearest Neighbors (KNN)",
                "Support Vector Machine (SVM)",
                "Random Forest"
            ])

        # ── COLUMN SELECTION ────────────────────────
        st.markdown("#### 3. Select Columns")

        col1, col2 = st.columns(2)
        with col1:
            x_cols = st.multiselect("Independent columns (X):", options=all_columns)
        with col2:
            y_col = st.selectbox("Dependent column (Y / Target):", options=[c for c in all_columns if c not in x_cols])

        # ── TEST SIZE ───────────────────────────────
        st.markdown("#### 4. Train-Test Split")
        test_size = st.slider("Test size (%):", min_value=10, max_value=40, value=20, step=5)
        st.caption(f"Training on **{100 - test_size}%** of data, testing on **{test_size}%**. `random_state=42` is used for reproducibility.")

        # ── TRAIN BUTTON ────────────────────────────
        st.markdown("---")
        train_btn = st.button("🚀 Train Model", use_container_width=True)

        if train_btn:
            if not x_cols:
                st.error("Please select at least one independent column (X).")
            elif not y_col:
                st.error("Please select a dependent column (Y).")
            elif y_col in x_cols:
                st.error("Dependent column (Y) cannot also be an independent column (X).")
            else:
                try:
                    # Prepare data
                    data = cleaned_df[x_cols + [y_col]].dropna()
                    X = data[x_cols]
                    y = data[y_col]

                    # Encode non-numeric X columns
                    X = pd.get_dummies(X)

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size / 100, random_state=42
                    )

                    st.markdown("---")
                    st.subheader("📊 Results")

                    # ────────────────────────────────
                    # REGRESSION MODELS
                    # ────────────────────────────────
                    if task_type == "Regression":

                        if model_choice in ["Linear Regression", "Multiple Linear Regression"]:
                            model = LinearRegression()
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            r2 = r2_score(y_test, y_pred)
                            mse = mean_squared_error(y_test, y_pred)

                            st.success(f"**Model:** {model_choice}  |  `random_state=42`")
                            c1, c2 = st.columns(2)
                            c1.metric("R² Score", f"{r2:.4f}")
                            c2.metric("Mean Squared Error", f"{mse:.4f}")

                            if len(x_cols) == 1:
                                st.markdown(f"**Equation:** y = {model.coef_[0]:.4f}x + {model.intercept_:.4f}")

                        elif model_choice == "Polynomial Regression":
                            # Only use numeric X columns — poly on dummies causes memory explosion
                            numeric_x = [c for c in x_cols if pd.api.types.is_numeric_dtype(cleaned_df[c])]
                            if not numeric_x:
                                st.error("Polynomial Regression requires at least one numeric independent column.")
                            else:
                                if len(numeric_x) < len(x_cols):
                                    st.warning(f"Non-numeric columns excluded from Polynomial Regression: {list(set(x_cols) - set(numeric_x))}")

                                X_train_p = X_train[numeric_x]
                                X_test_p = X_test[numeric_x]

                                st.write("🔍 Searching best degree (1–5)...")
                                d = {}
                                for i in range(1, 6):
                                    poly = PolynomialFeatures(degree=i, include_bias=False)
                                    X_tr = poly.fit_transform(X_train_p)
                                    X_te = poly.transform(X_test_p)
                                    m = LinearRegression()
                                    m.fit(X_tr, y_train)
                                    d[i] = r2_score(y_test, m.predict(X_te))

                                best_degree = max(d, key=d.get)

                                st.write(f"🏋️ Fitting final model with degree={best_degree}...")
                                poly = PolynomialFeatures(degree=best_degree, include_bias=False)
                                X_train_poly = poly.fit_transform(X_train_p)
                                X_test_poly = poly.transform(X_test_p)
                                model = LinearRegression()
                                model.fit(X_train_poly, y_train)
                                y_pred = model.predict(X_test_poly)
                                r2 = r2_score(y_test, y_pred)
                                mse = mean_squared_error(y_test, y_pred)

                                st.success(f"**Best Degree:** {best_degree} (searched 1–5, picked highest R²)  |  `random_state=42`")
                                c1, c2 = st.columns(2)
                                c1.metric("R² Score", f"{r2:.4f}")
                                c2.metric("Mean Squared Error", f"{mse:.4f}")

                                results_df = pd.DataFrame({
                                    "Degree": list(d.keys()),
                                    "R² Score": [round(v, 4) for v in d.values()]
                                })
                                st.markdown("**Degree Search Results:**")
                                st.dataframe(results_df, use_container_width=True)

                    # ────────────────────────────────
                    # CLASSIFICATION MODELS
                    # ────────────────────────────────
                    else:
                        if model_choice == "Decision Tree":
                            st.write("🔍 Searching best max_depth (1–20)...")
                            d = {}
                            for i in range(1, 21):
                                m = DecisionTreeClassifier(max_depth=i, criterion="entropy", random_state=42)
                                m.fit(X_train, y_train)
                                d[i] = accuracy_score(y_test, m.predict(X_test))

                            best_depth = max(d, key=d.get)

                            st.write(f"🏋️ Fitting final model with max_depth={best_depth}...")
                            model = DecisionTreeClassifier(max_depth=best_depth, criterion="entropy", random_state=42)
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            acc = accuracy_score(y_test, y_pred)
                            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

                            st.success(f"**Best max_depth:** {best_depth} (searched 1–20, picked highest accuracy)  |  `random_state=42`")
                            c1, c2 = st.columns(2)
                            c1.metric("Accuracy", f"{acc:.4f}")
                            c2.metric("F1 Score (weighted)", f"{f1:.4f}")

                            results_df = pd.DataFrame({
                                "Max Depth": list(d.keys()),
                                "Accuracy": [round(v, 4) for v in d.values()]
                            })
                            st.markdown("**Depth Search Results:**")
                            st.dataframe(results_df, use_container_width=True)

                        elif model_choice == "K-Nearest Neighbors (KNN)":
                            st.write("🔍 Searching best n_neighbors (1–20)...")
                            d = {}
                            for i in range(1, 21):
                                m = KNeighborsClassifier(n_neighbors=i)
                                m.fit(X_train, y_train)
                                d[i] = accuracy_score(y_test, m.predict(X_test))

                            best_k = max(d, key=d.get)

                            st.write(f"🏋️ Fitting final model with n_neighbors={best_k}...")
                            model = KNeighborsClassifier(n_neighbors=best_k)
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            acc = accuracy_score(y_test, y_pred)
                            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

                            st.success(f"**Best n_neighbors:** {best_k} (searched 1–20, picked highest accuracy)")
                            c1, c2 = st.columns(2)
                            c1.metric("Accuracy", f"{acc:.4f}")
                            c2.metric("F1 Score (weighted)", f"{f1:.4f}")

                            results_df = pd.DataFrame({
                                "K": list(d.keys()),
                                "Accuracy": [round(v, 4) for v in d.values()]
                            })
                            st.markdown("**K Search Results:**")
                            st.dataframe(results_df, use_container_width=True)

                        elif model_choice == "Support Vector Machine (SVM)":
                            from sklearn.svm import SVC
                            from sklearn.preprocessing import StandardScaler

                            st.write("🔍 Searching best C (0.01 → 100)...")
                            scaler = StandardScaler()
                            X_train_scaled = scaler.fit_transform(X_train)
                            X_test_scaled = scaler.transform(X_test)

                            d = {}
                            for C in [0.01, 0.1, 1, 10, 100]:
                                m = SVC(C=C, kernel="rbf", random_state=42)
                                m.fit(X_train_scaled, y_train)
                                d[C] = accuracy_score(y_test, m.predict(X_test_scaled))

                            best_C = max(d, key=d.get)

                            st.write(f"🏋️ Fitting final model with C={best_C}...")
                            model = SVC(C=best_C, kernel="rbf", random_state=42)
                            model.fit(X_train_scaled, y_train)
                            y_pred = model.predict(X_test_scaled)
                            acc = accuracy_score(y_test, y_pred)
                            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

                            st.success(f"**Best C:** {best_C} (searched [0.01, 0.1, 1, 10, 100], picked highest accuracy)  |  `random_state=42`")
                            st.caption("Features auto-scaled with StandardScaler before training.")
                            c1, c2 = st.columns(2)
                            c1.metric("Accuracy", f"{acc:.4f}")
                            c2.metric("F1 Score (weighted)", f"{f1:.4f}")

                            results_df = pd.DataFrame({
                                "C": list(d.keys()),
                                "Accuracy": [round(v, 4) for v in d.values()]
                            })
                            st.markdown("**C Search Results:**")
                            st.dataframe(results_df, use_container_width=True)

                        elif model_choice == "Random Forest":
                            st.write("🔍 Searching best n_estimators (10 → 300)...")
                            d = {}
                            for n in [10, 50, 100, 150, 200, 300]:
                                m = RandomForestClassifier(n_estimators=n, criterion="entropy", random_state=42)
                                m.fit(X_train, y_train)
                                d[n] = accuracy_score(y_test, m.predict(X_test))

                            best_n = max(d, key=d.get)

                            st.write(f"🏋️ Fitting final model with n_estimators={best_n}...")
                            model = RandomForestClassifier(n_estimators=best_n, criterion="entropy", random_state=42)
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            acc = accuracy_score(y_test, y_pred)
                            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

                            st.success(f"**Best n_estimators:** {best_n} (searched [10,50,100,150,200,300], picked highest accuracy)  |  `random_state=42`")
                            c1, c2 = st.columns(2)
                            c1.metric("Accuracy", f"{acc:.4f}")
                            c2.metric("F1 Score (weighted)", f"{f1:.4f}")

                            results_df = pd.DataFrame({
                                "n_estimators": list(d.keys()),
                                "Accuracy": [round(v, 4) for v in d.values()]
                            })
                            st.markdown("**Estimator Search Results:**")
                            st.dataframe(results_df, use_container_width=True)

                        # Classification report (all classifiers)
                        st.markdown("**Classification Report:**")
                        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                        st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

                        # Confusion Matrix
                        st.markdown("**Confusion Matrix:**")
                        import numpy as np
                        cm = confusion_matrix(y_test, y_pred)
                        labels = sorted(y_test.unique().tolist())

                        fig, ax = plt.subplots(figsize=(6, 4))
                        sns.heatmap(
                            cm,
                            annot=True,
                            fmt="d",
                            cmap="Blues",
                            xticklabels=labels,
                            yticklabels=labels,
                            ax=ax
                        )
                        ax.set_xlabel("Predicted")
                        ax.set_ylabel("Actual")
                        ax.set_title(f"Confusion Matrix — {model_choice}")
                        st.pyplot(fig)
                        plt.close(fig)

                # ── SAVE MODEL TO SESSION STATE ─────────────
                    model_label = f"Model {len(st.session_state.all_trained_models) + 1}: {model_choice} → {y_col}"
                    model_entry = {
                        "label": model_label,
                        "model": model,
                        "y_test": y_test,
                        "y_pred": y_pred,
                        "meta": {
                            "x_cols": x_cols,
                            "y_col": y_col,
                            "task_type": task_type,
                            "model_choice": model_choice,
                            "dummy_columns": X.columns.tolist(),
                            "original_x_dtypes": {col: str(cleaned_df[col].dtype) for col in x_cols},
                            "scaler": scaler if model_choice == "Support Vector Machine (SVM)" else None,
                            "poly": poly if model_choice == "Polynomial Regression" else None,
                            "poly_numeric_x": numeric_x if model_choice == "Polynomial Regression" else None,
                        }
                    }
                    st.session_state.all_trained_models.append(model_entry)
                    st.session_state.trained_model = model
                    st.session_state.training_meta = model_entry["meta"]
                    st.session_state.training_mode = False
                
                except Exception as e:
                    st.error(f"Training failed: {e}")


# ─────────────────────────────────────────────
    # REPORTS SECTION
    # ─────────────────────────────────────────────
    if "active_report_idx" in st.session_state and st.session_state.active_report_idx is not None:
        idx = st.session_state.active_report_idx
        entry = st.session_state.all_trained_models[idx]
        meta = entry["meta"]
        y_test = entry["y_test"]
        y_pred = entry["y_pred"]

        st.markdown("---")
        st.subheader(f"📋 Report — {entry['label']}")

        if meta["task_type"] == "Regression":
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            c1, c2 = st.columns(2)
            c1.metric("R² Score", f"{r2:.4f}")
            c2.metric("Mean Squared Error", f"{mse:.4f}")
        else:
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            c1, c2 = st.columns(2)
            c1.metric("Accuracy", f"{acc:.4f}")
            c2.metric("F1 Score (weighted)", f"{f1:.4f}")

            st.markdown("**Classification Report:**")
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

            st.markdown("**Confusion Matrix:**")
            import numpy as np
            cm = confusion_matrix(y_test, y_pred)
            labels = sorted(pd.Series(y_test).unique().tolist())
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                x=[str(l) for l in labels],
                y=[str(l) for l in labels],
                color_continuous_scale="Blues",
                labels=dict(x="Predicted", y="Actual"),
                title=f"Confusion Matrix — {meta['model_choice']}"
            )
            fig_cm.update_layout(template="plotly_white")
            st.plotly_chart(fig_cm, use_container_width=True)

        if st.button("✖ Close Report", use_container_width=True):
            st.session_state.active_report_idx = None
            st.rerun()

    # ─────────────────────────────────────────────
    # TRAINED MODELS SECTION
    # ─────────────────────────────────────────────
    if st.session_state.all_trained_models:
        st.markdown("---")
        st.subheader("🏆 Trained Models")

        if "active_report_idx" not in st.session_state:
            st.session_state.active_report_idx = None

        for i, entry in enumerate(st.session_state.all_trained_models):
            m = entry["meta"]
            with st.expander(f"**{entry['label']}**", expanded=(i == len(st.session_state.all_trained_models) - 1)):
                c1, c2, c3 = st.columns(3)
                c1.markdown(f"**Task:** {m['task_type']}")
                c2.markdown(f"**Target:** {m['y_col']}")
                c3.markdown(f"**Features:** {', '.join(m['x_cols'])}")
                if st.button(f"📋 Get Report", key=f"report_btn_{i}", use_container_width=True):
                    st.session_state.active_report_idx = i
                    st.rerun()

        if st.button("➕ Train Another Model", use_container_width=True):
            st.session_state.training_mode = True
            st.rerun()

# ─────────────────────────────────────────────
    # MODEL COMPARISON SECTION
    # ─────────────────────────────────────────────
    if len(st.session_state.all_trained_models) > 1:
        st.markdown("---")
        st.subheader("⚖️ Model Comparison")
        st.caption("Comparison of all trained models based on their evaluation metrics.")

        comparison_rows = []
        for entry in st.session_state.all_trained_models:
            m = entry["meta"]
            y_test = entry["y_test"]
            y_pred = entry["y_pred"]
            row = {
                "Model": entry["label"],
                "Task": m["task_type"],
                "Target": m["y_col"],
                "Features": ", ".join(m["x_cols"]),
            }
            if m["task_type"] == "Regression":
                row["R² Score"]  = round(r2_score(y_test, y_pred), 4)
                row["MSE"]       = round(mean_squared_error(y_test, y_pred), 4)
                row["Accuracy"]  = "—"
                row["F1 Score"]  = "—"
            else:
                row["Accuracy"]  = round(accuracy_score(y_test, y_pred), 4)
                row["F1 Score"]  = round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4)
                row["R² Score"]  = "—"
                row["MSE"]       = "—"
            comparison_rows.append(row)

        comp_df = pd.DataFrame(comparison_rows)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        # Bar chart — accuracy for classifiers or R² for regressors
        numeric_entries = [e for e in st.session_state.all_trained_models if e["meta"]["task_type"] == "Classification"]
        reg_entries     = [e for e in st.session_state.all_trained_models if e["meta"]["task_type"] == "Regression"]

        if len(numeric_entries) > 1:
            st.markdown("**Classifier Accuracy Comparison:**")
            clf_data = pd.DataFrame({
                "Model": [e["label"] for e in numeric_entries],
                "Accuracy": [round(accuracy_score(e["y_test"], e["y_pred"]), 4) for e in numeric_entries],
                "F1 Score": [round(f1_score(e["y_test"], e["y_pred"], average="weighted", zero_division=0), 4) for e in numeric_entries],
            })
            fig_cmp = px.bar(
                clf_data.melt(id_vars="Model", var_name="Metric", value_name="Score"),
                x="Model", y="Score", color="Metric", barmode="group",
                template="plotly_white", title="Classifier Comparison — Accuracy & F1",
                color_discrete_sequence=px.colors.qualitative.Bold,
                text_auto=".4f"
            )
            fig_cmp.update_layout(xaxis_tickangle=-20)
            st.plotly_chart(fig_cmp, use_container_width=True)

        if len(reg_entries) > 1:
            st.markdown("**Regressor R² Comparison:**")
            reg_data = pd.DataFrame({
                "Model": [e["label"] for e in reg_entries],
                "R² Score": [round(r2_score(e["y_test"], e["y_pred"]), 4) for e in reg_entries],
                "MSE": [round(mean_squared_error(e["y_test"], e["y_pred"]), 4) for e in reg_entries],
            })
            fig_reg = px.bar(
                reg_data.melt(id_vars="Model", var_name="Metric", value_name="Score"),
                x="Model", y="Score", color="Metric", barmode="group",
                template="plotly_white", title="Regressor Comparison — R² & MSE",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                text_auto=".4f"
            )
            fig_reg.update_layout(xaxis_tickangle=-20)
            st.plotly_chart(fig_reg, use_container_width=True)

    # ─────────────────────────────────────────────
    # PREDICTION SECTION
    # ─────────────────────────────────────────────
    if st.session_state.all_trained_models:
        st.markdown("---")
        st.subheader("🔮 Make a Prediction")

        model_labels = [e["label"] for e in st.session_state.all_trained_models]
        selected_label = st.selectbox("Select model to predict with:", model_labels, key="pred_model_select")
        selected_entry = next(e for e in st.session_state.all_trained_models if e["label"] == selected_label)
        meta = selected_entry["meta"]
        model_obj = selected_entry["model"]

        st.caption(f"Model: **{meta['model_choice']}** | Target: **{meta['y_col']}**")

        input_vals = {}
        cols_per_row = 3
        x_cols_list = meta["x_cols"]

        for i in range(0, len(x_cols_list), cols_per_row):
            row_cols = st.columns(cols_per_row)
            for j, col_name in enumerate(x_cols_list[i:i + cols_per_row]):
                dtype = meta["original_x_dtypes"][col_name]
                with row_cols[j]:
                    if "int" in dtype or "float" in dtype:
                        col_min = float(cleaned_df[col_name].min())
                        col_max = float(cleaned_df[col_name].max())
                        col_mean = float(cleaned_df[col_name].mean())
                        input_vals[col_name] = st.number_input(
                            label=col_name,
                            min_value=col_min,
                            max_value=col_max,
                            value=round(col_mean, 4),
                            help=f"Range: {col_min} – {col_max}",
                            key=f"pred_input_{col_name}_{selected_label}"
                        )
                    else:
                        unique_vals = cleaned_df[col_name].dropna().unique().tolist()
                        input_vals[col_name] = st.selectbox(
                            label=col_name,
                            options=sorted([str(v) for v in unique_vals]),
                            key=f"pred_select_{col_name}_{selected_label}"
                        )

        if st.button("🎯 Predict", use_container_width=True):
            try:
                input_df = pd.DataFrame([input_vals])

                if meta["model_choice"] == "Polynomial Regression":
                    input_numeric = input_df[meta["poly_numeric_x"]].astype(float)
                    input_processed = meta["poly"].transform(input_numeric)
                elif meta["model_choice"] == "Support Vector Machine (SVM)":
                    input_dummies = pd.get_dummies(input_df)
                    input_aligned = input_dummies.reindex(columns=meta["dummy_columns"], fill_value=0)
                    input_processed = meta["scaler"].transform(input_aligned)
                else:
                    input_dummies = pd.get_dummies(input_df)
                    input_processed = input_dummies.reindex(columns=meta["dummy_columns"], fill_value=0)

                prediction = model_obj.predict(input_processed)
                result = prediction[0]

                if meta["task_type"] == "Regression":
                    st.success(f"### 📈 Predicted {meta['y_col']}: `{round(float(result), 4)}`")
                else:
                    st.success(f"### 🏷️ Predicted {meta['y_col']}: `{result}`")

                with st.expander("📋 Input Summary"):
                    st.dataframe(pd.DataFrame([input_vals]), use_container_width=True)

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    # ─────────────────────────────────────────────
    # GET CODE SECTION
    # ─────────────────────────────────────────────
    if st.session_state.all_trained_models:
        st.markdown("---")
        st.subheader("📄 Get Training Code")
        st.caption("Ready-to-run Python code that reproduces the selected training session.")

        model_labels = [e["label"] for e in st.session_state.all_trained_models]
        selected_label_code = st.selectbox("Select model to get code for:", model_labels, key="code_model_select")
        selected_entry_code = next(e for e in st.session_state.all_trained_models if e["label"] == selected_label_code)
        meta = selected_entry_code["meta"]
        model_obj = selected_entry_code["model"]

        if st.button("🧾 Generate Code", use_container_width=True):
            model_choice = meta["model_choice"]
            task_type    = meta["task_type"]
            x_cols       = meta["x_cols"]
            y_col        = meta["y_col"]

            if model_choice in ["Linear Regression", "Multiple Linear Regression"]:
                model_init = "LinearRegression()"
                best_param_comment = ""
            elif model_choice == "Polynomial Regression":
                degree = meta["poly"].degree
                model_init = "LinearRegression()"
                best_param_comment = f"# Best degree found: {degree}"
            elif model_choice == "Decision Tree":
                depth = model_obj.max_depth
                model_init = f"DecisionTreeClassifier(max_depth={depth}, criterion='entropy', random_state=42)"
                best_param_comment = f"# Best max_depth found: {depth}"
            elif model_choice == "K-Nearest Neighbors (KNN)":
                k = model_obj.n_neighbors
                model_init = f"KNeighborsClassifier(n_neighbors={k})"
                best_param_comment = f"# Best n_neighbors found: {k}"
            elif model_choice == "Support Vector Machine (SVM)":
                c_val = model_obj.C
                model_init = f"SVC(C={c_val}, kernel='rbf', random_state=42)"
                best_param_comment = f"# Best C found: {c_val} | kernel=rbf"
            elif model_choice == "Random Forest":
                n_est = model_obj.n_estimators
                model_init = f"RandomForestClassifier(n_estimators={n_est}, criterion='entropy', random_state=42)"
                best_param_comment = f"# Best n_estimators found: {n_est}"

            imports = """import pandas as pd\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import {metrics}\n""".format(
                metrics="r2_score, mean_squared_error" if task_type == "Regression"
                        else "accuracy_score, f1_score, classification_report"
            )
            if model_choice in ["Linear Regression", "Multiple Linear Regression"]:
                imports += "from sklearn.linear_model import LinearRegression\n"
            elif model_choice == "Polynomial Regression":
                imports += "from sklearn.linear_model import LinearRegression\nfrom sklearn.preprocessing import PolynomialFeatures\n"
            elif model_choice == "Decision Tree":
                imports += "from sklearn.tree import DecisionTreeClassifier\n"
            elif model_choice == "K-Nearest Neighbors (KNN)":
                imports += "from sklearn.neighbors import KNeighborsClassifier\n"
            elif model_choice == "Support Vector Machine (SVM)":
                imports += "from sklearn.svm import SVC\nfrom sklearn.preprocessing import StandardScaler\n"
            elif model_choice == "Random Forest":
                imports += "from sklearn.ensemble import RandomForestClassifier\n"

            x_cols_repr = repr(x_cols)
            preprocess_block = f"\n# ── Load Data ──\ndf = pd.read_csv('your_file.csv')\n\n# ── Select columns ──\nX = df[{x_cols_repr}]\ny = df['{y_col}']\n"

            if model_choice == "Polynomial Regression":
                numeric_x = repr(meta["poly_numeric_x"])
                degree = meta["poly"].degree
                preprocess_block += f"\nX = X[{numeric_x}].astype(float)\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n{best_param_comment}\npoly = PolynomialFeatures(degree={degree}, include_bias=False)\nX_train = poly.fit_transform(X_train)\nX_test  = poly.transform(X_test)\n"
            elif model_choice == "Support Vector Machine (SVM)":
                preprocess_block += f"\nX = pd.get_dummies(X)\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\nscaler = StandardScaler()\nX_train = scaler.fit_transform(X_train)\nX_test  = scaler.transform(X_test)\n"
            else:
                preprocess_block += "\nX = pd.get_dummies(X)\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"

            train_block = f"\n# ── Train Model ──\n{best_param_comment}\nmodel = {model_init}\nmodel.fit(X_train, y_train)\ny_pred = model.predict(X_test)\n"
            metrics_block = "\n# ── Evaluate ──\nprint('R² Score :', r2_score(y_test, y_pred))\nprint('MSE      :', mean_squared_error(y_test, y_pred))\n" if task_type == "Regression" else "\n# ── Evaluate ──\nprint('Accuracy :', accuracy_score(y_test, y_pred))\nprint('F1 Score :', f1_score(y_test, y_pred, average='weighted', zero_division=0))\nprint(classification_report(y_test, y_pred, zero_division=0))\n"

            full_code = imports + preprocess_block + train_block + metrics_block
            st.code(full_code, language="python")
            st.download_button(
                label="⬇️ Download as .py file",
                data=full_code,
                file_name=f"{model_choice.replace(' ', '_').lower()}_training.py",
                mime="text/plain",
                use_container_width=True
            )

    # ─────────────────────────────────────────────
    # VISUALIZATION SECTION
    # ─────────────────────────────────────────────
    if st.session_state.all_trained_models:
        st.markdown("---")
        st.subheader("📊 Visualizations")

        all_cols = cleaned_df.columns.tolist()
        numeric_cols_viz = cleaned_df.select_dtypes(include="number").columns.tolist()

        COLOR_PALETTES = {
            "Default Blue": px.colors.sequential.Blues_r,
            "Viridis": px.colors.sequential.Viridis,
            "Plasma": px.colors.sequential.Plasma,
            "Sunset": px.colors.sequential.Sunset,
            "Teal": px.colors.sequential.Teal,
            "Red-Orange": px.colors.sequential.Reds,
            "Green": px.colors.sequential.Greens,
            "Rainbow": px.colors.qualitative.Plotly,
            "Pastel": px.colors.qualitative.Pastel,
            "Bold": px.colors.qualitative.Bold,
        }

        st.markdown("#### Configure a Chart")

        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            graph_type = st.selectbox("Chart type:", [
                "Scatter Plot", "Histogram", "Pie Chart",
                "Bar Chart", "Line Chart", "Heatmap", "Box Plot"
            ], key="viz_graph_type")
        with fc2:
            model_labels_viz = [e["label"] for e in st.session_state.all_trained_models]
            chart_model_label = st.selectbox("Associated model:", model_labels_viz, key="viz_model")
        with fc3:
            selected_palette = st.selectbox("Color palette:", list(COLOR_PALETTES.keys()), key="viz_palette")

        chosen_colors = COLOR_PALETTES[selected_palette]

        # Dynamic controls — outside form so they update live
        if graph_type == "Scatter Plot":
            sc1, sc2, sc3 = st.columns(3)
            x_axis    = sc1.selectbox("X axis:", all_cols, key="viz_sc_x")
            y_axis    = sc2.selectbox("Y axis:", numeric_cols_viz, key="viz_sc_y")
            color_col = sc3.selectbox("Color by (optional):", ["None"] + all_cols, key="viz_sc_c")

        elif graph_type == "Histogram":
            h1, h2 = st.columns(2)
            x_axis = h1.selectbox("Column:", numeric_cols_viz, key="viz_hist_x")
            nbins  = h2.slider("Number of bins:", 5, 100, 20, key="viz_hist_bins")

        elif graph_type == "Pie Chart":
            p1, p2 = st.columns(2)
            pie_names  = p1.selectbox("Category column:", all_cols, key="viz_pie_names")
            pie_values = p2.selectbox("Value column:", numeric_cols_viz, key="viz_pie_vals")

        elif graph_type == "Bar Chart":
            b1, b2, b3 = st.columns(3)
            x_axis  = b1.selectbox("X axis (category):", all_cols, key="viz_bar_x")
            y_axis  = b2.selectbox("Y axis (value):", numeric_cols_viz, key="viz_bar_y")
            bar_agg = b3.selectbox("Aggregation:", ["sum", "mean", "count"], key="viz_bar_agg")

        elif graph_type == "Line Chart":
            l1, l2, l3 = st.columns(3)
            x_axis    = l1.selectbox("X axis:", all_cols, key="viz_line_x")
            y_axis    = l2.selectbox("Y axis:", numeric_cols_viz, key="viz_line_y")
            color_col = l3.selectbox("Color by (optional):", ["None"] + all_cols, key="viz_line_c")

        elif graph_type == "Heatmap":
            hm_cols = st.multiselect(
                "Select numeric columns for heatmap:",
                numeric_cols_viz,
                default=numeric_cols_viz[:min(8, len(numeric_cols_viz))],
                key="viz_hm_cols"
            )
        
        elif graph_type == "Box Plot":
            bx1, bx2 = st.columns(2)
            box_y = bx1.selectbox("Value column (Y):", numeric_cols_viz, key="viz_box_y")
            box_x = bx2.selectbox("Group by (X, optional):", ["None"] + all_cols, key="viz_box_x")

        if st.button("📈 Plot Graph", use_container_width=True):
            try:
                fig = None
                chart_title = f"{graph_type} | {chart_model_label}"
                # Use first color from palette as single color for non-categorical charts
                single_color = chosen_colors[0] if isinstance(chosen_colors[0], str) else "#636EFA"

                if graph_type == "Scatter Plot":
                    color = color_col if color_col != "None" else None
                    fig = px.scatter(cleaned_df, x=x_axis, y=y_axis, color=color,
                                     title=chart_title, template="plotly_white",
                                     color_discrete_sequence=chosen_colors)

                elif graph_type == "Histogram":
                    fig = px.histogram(cleaned_df, x=x_axis, nbins=nbins,
                                       title=chart_title, template="plotly_white",
                                       color_discrete_sequence=chosen_colors)

                elif graph_type == "Pie Chart":
                    pie_df = cleaned_df.groupby(pie_names)[pie_values].sum().reset_index()
                    fig = px.pie(pie_df, names=pie_names, values=pie_values,
                                 title=chart_title,
                                 color_discrete_sequence=chosen_colors)

                elif graph_type == "Bar Chart":
                    if bar_agg == "sum":
                        bar_df = cleaned_df.groupby(x_axis)[y_axis].sum().reset_index()
                    elif bar_agg == "mean":
                        bar_df = cleaned_df.groupby(x_axis)[y_axis].mean().reset_index()
                    else:
                        bar_df = cleaned_df.groupby(x_axis)[y_axis].count().reset_index()
                    fig = px.bar(bar_df, x=x_axis, y=y_axis,
                                 title=chart_title, template="plotly_white",
                                 color_discrete_sequence=chosen_colors)

                elif graph_type == "Line Chart":
                    color = color_col if color_col != "None" else None
                    fig = px.line(cleaned_df.sort_values(x_axis), x=x_axis, y=y_axis,
                                  color=color, title=chart_title, template="plotly_white",
                                  color_discrete_sequence=chosen_colors)

                elif graph_type == "Heatmap":
                    if len(hm_cols) < 2:
                        st.warning("Select at least 2 columns for a heatmap.")
                    else:
                        corr = cleaned_df[hm_cols].corr().round(2)
                        fig = px.imshow(corr, text_auto=True,
                                        color_continuous_scale=selected_palette.lower().replace(" ", "_"),
                                        title=chart_title)
                        
                elif graph_type == "Box Plot":
                    x = box_x if box_x != "None" else None
                    fig = px.box(cleaned_df, x=x, y=box_y,
                                 title=chart_title, template="plotly_white",
                                 color=x,
                                 color_discrete_sequence=chosen_colors)

                if fig:
                    st.session_state.plotted_graphs.append({
                        "label": chart_title,
                        "fig": fig
                    })
                    st.rerun()  # scroll down to show new graph immediately

            except Exception as e:
                st.error(f"Plotting failed: {e}")

    # ─────────────────────────────────────────────
    # GRAPHS PLOTTED SECTION
    # ─────────────────────────────────────────────
    if st.session_state.plotted_graphs:
        st.markdown("---")
        st.subheader("🖼️ Graphs Plotted")
        st.caption(f"{len(st.session_state.plotted_graphs)} graph(s) plotted this session.")

        for i, graph in enumerate(st.session_state.plotted_graphs):
            with st.expander(f"**{graph['label']}**", expanded=True):
                st.plotly_chart(graph["fig"], use_container_width=True)

        if st.button("🗑️ Clear All Graphs", use_container_width=True):
            st.session_state.plotted_graphs = []
            st.rerun()