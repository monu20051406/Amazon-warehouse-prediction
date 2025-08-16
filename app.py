import streamlit as st
import pandas as pd
import os
import zipfile
import io 
from inventory.predictor import run_full_forecast
from inventory.tracking import inventory_ledger
from gsheets_io import read_df, write_df, append_df, upsert_df

# --- Backend paths to static files ---
UPLOADS_PATH = "data/uploads"
OUTPUTS_PATH = "data/outputs"

st.set_page_config(page_title="Vibrant Forecast", layout="wide")

creds = st.secrets["gcp_service_account"]

# --- Caching the Main Forecasting Function ---
@st.cache_data # CRITICAL: Caches the output to prevent re-running on every interaction
def cached_forecast(sales_file, inv_file, time_unit, prediction_choice):
    """Wrapper function to cache the results of the main forecast."""
    # The run_full_forecast function now accepts the file objects directly
    return run_full_forecast(
        sales_file=sales_file,
        inv_file=inv_file,
        time_unit=time_unit,
        prediction_choice=prediction_choice,
        creds = creds
    )

# --- Sidebar for page navigation ---
page = st.sidebar.radio("Select a Page", ["Vibrant Inventory Forecaster", "Inventory Tracking"])

# --- Main page navigation logic ---
if page == "Vibrant Inventory Forecaster":
    # Title and Subheader for Inventory Forecaster Page
    st.markdown("""
    <div class='vibrant-title' style='font-size:52px; font-weight:900; letter-spacing:-2px;
    background: linear-gradient(90deg, #405cfe 12%, #38b6ff 52%, #6ae3f9 88%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
    VIBRANT INVENTORY FORECASTER
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Upload weekly sales and inventory data to predict next week’s demand by ASIN and cluster.")
    
    # --- Upload Section ---
    st.markdown("### 🗂️ Upload Data Files")
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Sales File**: 
            - Go to <a href="https://sellercentral.amazon.in/ap/signin" target="_blank" class="seller-central-link">Seller Central</a> and open the left sidebar menu.
            - Go to Reports → Fulfillment → Sales
            - Download the <b>Customer Shipment Sales Report</b> for the last 7 days.
            - Upload the file here (CSV .csv format).
            """, unsafe_allow_html=True)
            sales_file = st.file_uploader("📊 Sales File (.csv)", type=["csv"], key="sales")
        with col2:
            st.markdown("""
            **Inventory File**: 
            - Go to <a href="https://sellercentral.amazon.in/ap/signin" target="_blank" class="seller-central-link">Seller Central</a> and open the left sidebar menu.
            - Go to Reports → Fulfillment → Inventory
            - Download the <b>Inventory Ledger Report</b> for last 7 days.
            - Upload the file here (CSV .csv format).
            """, unsafe_allow_html=True)
            inventory_file = st.file_uploader("🏬 Inventory File (.csv)", type="csv", key="inv")

    # --- Prediction Duration Section ---
    if sales_file and inventory_file:
        st.success("Files uploaded! Choose prediction duration and run the forecast.")

        st.markdown("### 📅 Choose Prediction Duration")
        col1, col2 = st.columns([1, 1])

        with col2:
            time_unit = st.selectbox(
                "Choose Time Unit",
                ["Week", "Month"],
                help="Select whether you want the prediction in weeks or months."
            )
        with col1:
            if time_unit == "Week":
                number_options = [1, 2, 3]
            else:
                number_options = [1, 2, 3, 4]
            prediction_choice = st.selectbox(
                "Make prediction for",
                number_options,
                help="Select the number of weeks or months for the prediction."
            )

        st.write(f"Prediction for {prediction_choice} {time_unit}{'s' if prediction_choice > 1 else ''}.")

        # === Forecast Button ===
        st.markdown("### 📈 Forecast Demand")
        if st.button("▶️ Run Forecast Now"):
            os.makedirs(UPLOADS_PATH, exist_ok=True)
            os.makedirs(OUTPUTS_PATH, exist_ok=True)

            # --- Save uploaded files
            sales_path = os.path.join(UPLOADS_PATH, sales_file.name)
            inventory_path = os.path.join(UPLOADS_PATH, inventory_file.name)

            with open(sales_path, "wb") as f: f.write(sales_file.getbuffer())
            with open(inventory_path, "wb") as f: f.write(inventory_file.getbuffer())

            with st.spinner("Processing your data and forecasting (~ 1-2 minutes)..."):
                    try:
                        # **Call the CACHED forecast function with the file OBJECTS**
                        maj_df, min_df, latest_date = cached_forecast(
                            sales_file,
                            inventory_file,
                            time_unit,
                            prediction_choice
                        )
                        # Save results to session state to persist them
                        st.session_state.maj_df = maj_df
                        st.session_state.min_df = min_df
                        st.session_state.latest_date = latest_date
                        st.session_state.prediction_choice = prediction_choice
                        st.session_state.time_unit = time_unit
                        
                        st.balloons()
                        st.success("Forecast complete! View and filter your reports below.")
                    except Exception as e:
                        st.error(f"Error during forecast: {e}")
                        st.stop()
    elif not (sales_file and inventory_file):
        st.info("Please upload both the Sales and Inventory files to proceed.")


    # --- Display Forecasts and Controls if they exist in session state ---
    if 'maj_df' in st.session_state and 'min_df' in st.session_state:
        maj_df = st.session_state.maj_df
        min_df = st.session_state.min_df
        latest_date = st.session_state.latest_date
        prediction_choice = st.session_state.prediction_choice
        time_unit = st.session_state.time_unit

        st.markdown("---")
        st.markdown("### 📋 Forecast Results")

        prediction_start_date = latest_date + pd.Timedelta(days=1)
        if time_unit == "Week":
            prediction_end_date = prediction_start_date + pd.Timedelta(days=7*prediction_choice - 1)
        else:  # Month
            prediction_end_date = prediction_start_date + pd.DateOffset(months=prediction_choice) - pd.Timedelta(days=1)
        st.info( f" Prediction period: **{prediction_start_date.strftime('%Y-%m-%d')}** to **{prediction_end_date.strftime('%Y-%m-%d')}**" )

        # --- Majority Group Forecast with Filters ---
        st.markdown("#### Majority Group Forecast")
        maj_col1, maj_col2 = st.columns(2)
        with maj_col1:
            asins_maj = st.multiselect(
                "Choose an ASIN",
                options=sorted(maj_df['ASIN'].unique()),
                default=None,
                key="maj_asin_filter"
            )
        with maj_col2:
            clusters_maj = st.multiselect(
                "Choose a Cluster",
                options=sorted(maj_df['Cluster'].unique()),
                default=None,
                key="maj_cluster_filter"
            )

        filtered_maj_df = maj_df.copy()
        if asins_maj:
            filtered_maj_df = filtered_maj_df[filtered_maj_df['ASIN'].isin(asins_maj)]
        if clusters_maj:
            filtered_maj_df = filtered_maj_df[filtered_maj_df['Cluster'].isin(clusters_maj)]
        
        st.dataframe(filtered_maj_df, use_container_width=True)
        
        # --- Minority Group Forecast with Filters ---
        st.markdown("#### Minority Group Forecast")
        min_col1, min_col2 = st.columns(2)
        with min_col1:
            asins_min = st.multiselect(
                "Choose an ASIN",
                options=sorted(min_df['ASIN'].unique()),
                default=None,
                key="min_asin_filter"
            )
        with min_col2:
            clusters_min = st.multiselect(
                "Choose a Cluster",
                options=sorted(min_df['Cluster'].unique()),
                default=None,
                key="min_cluster_filter"
            )

        filtered_min_df = min_df.copy()
        if asins_min:
            filtered_min_df = filtered_min_df[filtered_min_df['ASIN'].isin(asins_min)]
        if clusters_min:
            filtered_min_df = filtered_min_df[filtered_min_df['Cluster'].isin(clusters_min)]

        st.dataframe(filtered_min_df, use_container_width=True)
        
        st.markdown("---")

        # --- Create a zip file in memory ---
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            maj_excel_buffer = io.BytesIO()
            maj_df.to_excel(maj_excel_buffer, index=False)
            maj_excel_buffer.seek(0)

            min_excel_buffer = io.BytesIO()
            min_df.to_excel(min_excel_buffer, index=False)
            min_excel_buffer.seek(0)
            
            zipf.writestr("majority_forecast.xlsx", maj_excel_buffer.read())
            zipf.writestr("minority_forecast.xlsx", min_excel_buffer.read())
        
        zip_buffer.seek(0)

        # --- Download Button ---
        st.download_button(
            label="⬇️ Download Both Forecasts (ZIP)",
            data=zip_buffer,
            file_name=f"forecast_reports_{prediction_choice}_{time_unit}.zip",
            mime="application/zip"
        )


elif page == "Inventory Tracking":
    
    TAB_INV_SUMMARY = "inventory_summary"

    st.markdown("""
        <div class='vibrant-title' style='font-size:52px; font-weight:900; letter-spacing:-2px;
        background: linear-gradient(90deg, #405cfe 12%, #38b6ff 52%, #6ae3f9 88%); 
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
        VIBRANT INVENTORY TRACKING
        </div>
        """, unsafe_allow_html=True)
    st.markdown("### Track and Manage your inventory at Amazon warehouses.")
    st.markdown("""
                **Inventory File**: 
            - Go to <a href="https://sellercentral.amazon.in/ap/signin" target="_blank" class="seller-central-link">Seller Central</a> and open the left sidebar menu.
            - Go to Reports → Fulfillment → Inventory
            - Download the <b>Inventory Ledger Report</b> for last 7 days.
            - Upload the file here (CSV .csv format).""", unsafe_allow_html=True)
    st.markdown("---")

    try:
        master_df = read_df(TAB_INV_SUMMARY)
        if not master_df.empty and 'Last Order Date' in master_df.columns:
            master_df['Last Order Date'] = pd.to_datetime(master_df['Last Order Date'], errors='coerce')
        else:
            master_df = pd.DataFrame()
    except Exception as e:
        st.error(f"Error reading inventory summary from Google Sheets: {e}")
        master_df = pd.DataFrame()

    if not master_df.empty:
        latest_date = master_df['Last Order Date'].max()
        st.info(f" Latest inventory data is available up to: **{latest_date.strftime('%Y-%m-%d')}**")
    else:
        st.info("ℹ️ No inventory summary found. Upload a file to generate one.")

    inven_file = st.file_uploader("📦 Upload Inventory Ledger Report (.csv)", type="csv")

    if inven_file:
        if st.button("Merge & Update Inventory Summary", type="primary"):
            with st.spinner("Processing new file and updating summary..."):
                inventory_ledger(inven_file=inven_file, creds=creds)
                st.success("✅ Inventory summary updated successfully!")
                st.rerun()

    st.markdown("---")
    st.markdown("#### Current Inventory Summary")

    if not master_df.empty:
        inv_col1, inv_col2 = st.columns(2)
        with inv_col1:
            asins_inv = st.multiselect(
                "Choose an ASIN",
                options=sorted(master_df['ASIN'].astype(str).unique()),
                default=None,
                key="inv_asins_filter"
            )
        with inv_col2:
            warehouse_inv = st.multiselect(
                "Choose a Warehouse",
                options=sorted(master_df['Warehouse'].unique()),
                default=None,
                key="inv_warehouse_filter"
            )

        display_df = master_df.copy()      
        if asins_inv:
            display_df = display_df[display_df['ASIN'].isin(asins_inv)]
        if warehouse_inv:
            display_df = display_df[display_df['Warehouse'].isin(warehouse_inv)]
        
        display_df_formatted = display_df.copy()
        display_df_formatted['Last Order Date'] = display_df_formatted['Last Order Date'].dt.strftime('%Y-%m-%d')
        
        st.dataframe(display_df_formatted, use_container_width=True)

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            filtered_excel_buffer = io.BytesIO()
            display_df.to_excel(filtered_excel_buffer, index=False)
            filtered_excel_buffer.seek(0)
            zipf.writestr("inventory_summary_filtered.xlsx", filtered_excel_buffer.read())

            full_excel_buffer = io.BytesIO()
            master_df.to_excel(full_excel_buffer, index=False)
            full_excel_buffer.seek(0)
            zipf.writestr("inventory_summary_full.xlsx", full_excel_buffer.read())

        zip_buffer.seek(0)

        st.download_button(
            label="⬇️ Download Inventory Summary (ZIP)",
            data=zip_buffer,
            file_name=f"inventory_summary_{latest_date.strftime('%Y-%m-%d') if not master_df.empty else 'export'}.zip",
            mime="application/zip"
        )

    else:
        st.markdown("No data to display. Please upload a file.")
