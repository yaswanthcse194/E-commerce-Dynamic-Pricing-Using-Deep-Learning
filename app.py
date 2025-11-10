import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Dynamic Pricing", page_icon="🛒", layout="wide")
st.title("🛒 Dynamic Pricing Recommendation")

# Load data
df = pd.read_csv("retail_price.csv")

# Ensure datetime
if 'month_year' in df.columns:
    df['month_year'] = pd.to_datetime(df['month_year'], errors='coerce')

# Basic cleaning to mirror notebook assumptions
df = df.drop_duplicates()

# Feature engineering (aligned with notebook)
if 'month_year' in df.columns:
    df['month'] = df['month_year'].dt.month
    df['year'] = df['month_year'].dt.year
    # quarter support across pandas versions
    try:
        df['quarter'] = df['month_year'].dt.quarter
    except Exception:
        df['quarter'] = ((df['month'] - 1) // 3) + 1

# Competitor gaps if columns exist
for comp_col in ['comp_1', 'comp_2', 'comp_3']:
    if comp_col in df.columns and 'unit_price' in df.columns:
        gap_col = f"gap_{comp_col.replace('comp_', 'comp')}"
        # gap_comp1, gap_comp2, gap_comp3
        df[gap_col] = df['unit_price'] - df[comp_col]

# Monthly demand per product if possible
if {'product_id', 'month', 'qty'}.issubset(df.columns):
    df['monthly_demand'] = df.groupby(['product_id', 'month'])['qty'].transform('mean')

# Price lag by product if possible
if {'product_id', 'unit_price'}.issubset(df.columns):
    df = df.sort_values(['product_id'] + (['month_year'] if 'month_year' in df.columns else []))
    df['price_lag1'] = df.groupby('product_id')['unit_price'].shift(1)
    df['price_lag1'] = df['price_lag1'].fillna(df['unit_price'])

# Elasticity proxy
if {'unit_price', 'price_lag1', 'qty'}.issubset(df.columns):
    df['elasticity'] = (df['unit_price'] - df['price_lag1']).abs() / (df['qty'] + 1)

# Minimal theming
st.markdown(
    """
    <style>
    .metric-card {padding: 16px; border: 1px solid #e9ecef; border-radius: 10px; background: #ffffff;}
    .small-note {color:#6c757d; font-size: 12px;}
    .stAlert {margin-top: 0.5rem; margin-bottom: 0.5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Controls")
    st.caption("Filter a product and (optionally) a date to predict.")

    # UI: pick a product row to predict
    row_id = None
    selected_time = None
    if 'product_id' in df.columns:
        product_ids = df['product_id'].astype(str).unique().tolist()
        selected_product = st.selectbox("Product ID", options=product_ids)
        filtered = df[df['product_id'].astype(str) == selected_product]

        if len(filtered) == 0:
            st.stop()

        if 'month_year' in filtered.columns:
            times = filtered['month_year'].dt.strftime("%Y-%m-%d").fillna("NA").tolist()
            # Keep stable order
            selected_time = st.selectbox("Month/Year", options=times)
            try:
                if selected_time != "NA":
                    target = filtered[filtered['month_year'].dt.strftime("%Y-%m-%d") == selected_time]
                    target_row = target.iloc[0] if len(target) else filtered.iloc[0]
                else:
                    target_row = filtered.iloc[0]
            except Exception:
                target_row = filtered.iloc[0]
        else:
            target_row = filtered.iloc[0]
        row_id = target_row.name
    else:
        # Fallback: just use first row
        row_id = df.index[0]

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.caption("Model")
    st.write("RandomForest (joblib)")

st.markdown("#### Dataset")
with st.expander("Preview first 10 rows", expanded=False):
    st.dataframe(df.head(10), use_container_width=True)

# Features used during training (subset will be taken based on availability)
FEATURES = [
    'qty', 'monthly_demand', 'price_lag1', 'elasticity',
    'comp_1', 'comp_2', 'comp_3',
    'gap_comp1', 'gap_comp2', 'gap_comp3',
    'month', 'quarter', 'year'
]
FEATURES = [f for f in FEATURES if f in df.columns]

# Load model trained in notebook
model = joblib.load("price_model.pkl")

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📈 Insights", "ℹ️ About"])

with tab1:
    st.subheader("Predict Price")
    if row_id is None:
        st.error("No valid row selected for prediction.")
    else:
        row = df.loc[row_id]
        X = pd.DataFrame([row[FEATURES].values], columns=FEATURES)
        pred_price = float(model.predict(X)[0])
        current_price = float(row['unit_price']) if 'unit_price' in row else np.nan
        qty_val = float(row['qty']) if 'qty' in row else np.nan

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Predicted Price", f"₹{pred_price:,.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            if not np.isnan(current_price):
                delta = ((pred_price - current_price) / current_price) * 100 if current_price != 0 else np.nan
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Current Price", f"₹{current_price:,.2f}", f"{delta:+.2f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Current Price", "—")
                st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            est_profit = pred_price * qty_val if not np.isnan(qty_val) else np.nan
            est_profit_txt = f"₹{est_profit:,.0f}" if not np.isnan(est_profit) else "—"
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Estimated Profit", est_profit_txt)
            st.markdown('<span class="small-note">Approx: predicted price × qty</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.subheader("Product Insights")
    if 'product_id' in df.columns:
        pid = df.loc[row_id]['product_id'] if row_id is not None else None
        product_df = df[df['product_id'] == pid] if pid is not None else df.copy()
        # Compute predictions for the product set
        if len(FEATURES) > 0 and len(product_df) > 0:
            preds = model.predict(product_df[FEATURES])
            show_df = product_df.copy()
            show_df['predicted_price'] = preds

            c1, c2 = st.columns(2)
            with c1:
                st.caption("Actual vs Predicted (table)")
                columns_to_show = ['unit_price', 'predicted_price']
                if 'month_year' in show_df.columns:
                    columns_to_show = ['month_year'] + columns_to_show
                st.dataframe(show_df[columns_to_show].head(20), use_container_width=True)
            with c2:
                st.caption("Error summary")
                if 'unit_price' in show_df.columns:
                    err = (show_df['predicted_price'] - show_df['unit_price']).abs()
                    st.write("MAE:", float(np.nanmean(err)))
                else:
                    st.info("Unit price not available to compute error.")
    else:
        st.info("No product_id column to show product-specific insights.")

with tab3:
    st.subheader("About")
    st.write(
        "This app estimates a product's optimal selling price using a RandomForest model "
        "trained on engineered features such as demand proxies, price lags, and competitor gaps."
    )
