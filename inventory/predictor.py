import pandas as pd
import numpy as np
from datetime import timedelta
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import xgboost as xgb
from scipy.stats import linregress
import time
import math
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from gsheets_io import read_df, write_df, append_df, upsert_df

TAB_HIST            = "historical_sales_data_6jul_29jun"
TAB_CLUSTER_MAP     = "Cluster_Warehouse_Mapping"
TAB_COORDS          = "postal_code_coords"
TAB_COMBINED        = "combined_df"   
TAB_MAJORITY_DATA   = "majority_data_df"
TAB_MINORITY_DATA   = "minority_data_df"
TAB_TOP3 = "top_3_warehouses"

'''def read_uploaded_file(file):
    """Function to read an uploaded file as a dataframe"""
    if file.type == "text/csv":
        return pd.read_csv(file)  # Read as CSV
    elif file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        return pd.read_excel(file)  # Read as Excel
    else:
        raise ValueError("Unsupported file type")'''

def run_full_forecast(
    sales_file,
    inv_file,   
    time_unit: str,
    prediction_choice: int,
    creds
):
    if sales_file.type == "text/csv":
        new_sales_df = pd.read_csv(sales_file)
    else:
        new_sales_df = pd.read_excel(sales_file)

    if inv_file.type == "text/csv":
        inv_df = pd.read_csv(inv_file)
    else:
        inv_df = pd.read_excel(inv_file)

    hist_df = read_df(TAB_HIST, creds)
    cluster_wh_df = read_df(TAB_CLUSTER_MAP, creds)

    new_sales_df = new_sales_df.rename(columns={'FC': 'Warehouse ID', 'Shipment To Postal Code': 'Ship Postal Code'})
    
    columns_to_remove = [
        'Merchant SKU', 'FNSKU', 'Currency', 'Shipping Amount', 'Amazon Order Id', 'Gift Amount', 'Shipment To City', 'Shipment To State'
    ]
    new_sales_df = new_sales_df.drop(columns=columns_to_remove, errors='ignore')
    new_sales_df = new_sales_df.rename(columns={'FC': 'Warehouse ID', 'Shipment To Postal Code': 'Ship Postal Code'})
   
    remove_inv = [
        'MSKU', 'FNSKU', 'Title', 'Dispositition', 'Starting Warehouse Balance', 'In Transit Between Warehouses' , 'Amazon Order Id',
        'Receipts', 'Customer Shipments', 'Customer Returns', 'Vendor Returns', 'Warehouse Transfer In/Out', 'Found', 'Lost', 'Damaged', 'Disposed', 'Other Events', 'Unknown Events'
        ]
    inv_df = inv_df.drop(columns=remove_inv, errors='ignore')
    inv_df['Date'] = pd.to_datetime(inv_df['Date'], errors='coerce')
    inv_df['Date'] = inv_df['Date'].dt.date

        # ---------- FC to Pincode Mapping ----------
    fc_to_pincode = {
        'SGAA': 781132, 'SGAC': 781101, 'SPAB': 801103, 'DEX3': 110044, 'PNQ2': 110044, 'DEX8': 110044,
        'AMD2': 382220, 'SAME': 387570, 'DEL2': 122105, 'DEL4': 122503, 'DEL5': 122413, 'DED3': 122506,
        'DED5': 122103, 'SDEG': 122105, 'SDEB': 124108, 'XNRW': 124108, 'BLR4': 562149, 'BLR5': 562114,
        'BLR7': 562107, 'BLR8': 562149, 'BLX1': 563160, 'XSAJ': 562132, 'BOM5': 421302, 'BOM7': 421302,
        'ISK3': 421302, 'PNQ3': 410501, 'SBOB': 421101, 'XWAA': 421302, 'SBHF': 462030, 'SIDA': 453771,
        'ATX1': 141113, 'SATB': 141113, 'JPX1': 302016, 'JPX2': 303007, 'HYD8': 500108, 'HYD3': 500108,
        'SHYH': 500108, 'SHYB': 502279, 'XSAD': 502279, 'XSIP': 502279, 'MAA4': 601206, 'CJB1': 641201,
        'SMAB': 601103, 'SCJF': 641402, 'XSIR': 601103, 'LKO1': 226401, 'SLKD': 226401, 'CCX1': 711322,
        'CCX2': 711302, 'SCCE': 711313, 'XECP': 711401, 'PAX1': 800009, 'TQBF': 400092
    }

    # ---------- POSTAL CODE TO COORDS ----------
    try:
        coord_df = read_df(TAB_COORDS, creds)
        pin_to_coord = {int(row['Pincode']): (row['Latitude'], row['Longitude'])
                        for _, row in coord_df.iterrows()
                        if pd.notna(row.get('Pincode'))}
    except Exception:
        coord_df = pd.DataFrame(columns=['Pincode','Latitude','Longitude'])
        pin_to_coord = {}

    geolocator = Nominatim(user_agent="fc_mapper_v5")
    def geocode_pin(pin):
        try:
            location = geolocator.geocode(f"India {int(pin)}", timeout=10)
            if location:
                return (location.latitude, location.longitude)
        except Exception as e:
            print(f"Failed for {pin}: {e}")
        return (None, None)

    # ---------- 3. DEFINE THE WAREHOUSE REASSIGNMENT LOGIC ----------
    def get_reassigned_warehouse(row, pin_to_coord, fc_to_pincode):
        try:
            ship_postal = row['Ship Postal Code']
            original_fc = str(row['Warehouse ID']).strip()

            if pd.isna(ship_postal) or not original_fc:
                return original_fc

            ship_postal = int(ship_postal)
            ship_coord = pin_to_coord.get(ship_postal)
            original_fc_pin = fc_to_pincode.get(original_fc)

            if not all([ship_coord, original_fc_pin]):
                return original_fc

            original_fc_coord = pin_to_coord.get(original_fc_pin)
            if not original_fc_coord:
                return original_fc

            distance = geodesic(ship_coord, original_fc_coord).km

            if distance <= 300:
                return original_fc
            else:
                min_dist = float('inf')
                nearest_fc_code = original_fc

                for fc_code, fc_pin in fc_to_pincode.items():
                    fc_coord = pin_to_coord.get(fc_pin)
                    if fc_coord:
                        dist = geodesic(ship_coord, fc_coord).km
                        if dist < min_dist:
                            min_dist = dist
                            nearest_fc_code = fc_code
                return nearest_fc_code
        except (ValueError, TypeError):
            return str(row['Warehouse ID']).strip() 
        
    # ---------- 4. COMBINE DATA & APPLY LOGIC ----------
    cluster_wh_df.columns = cluster_wh_df.columns.str.strip()
    warehouse_to_cluster = {str(wh).strip(): str(row['Cluster']).strip() for _, row in cluster_wh_df.iterrows() for wh in row[1:] if pd.notna(wh)}

    hist_df['Customer Shipment Date'] = pd.to_datetime(hist_df['Customer Shipment Date'], errors='coerce').dt.date
    hist_df.drop_duplicates( subset=[ "Customer Shipment Date", "ASIN", "Quantity", "Product Amount", "Warehouse ID", "Cluster", "Ship Postal Code", "Sales"], keep="last", inplace=True)

    new_sales_df['Customer Shipment Date'] = pd.to_datetime(new_sales_df['Customer Shipment Date'], errors='coerce').dt.date
    new_sales_df['Ship Postal Code'] = pd.to_numeric(new_sales_df['Ship Postal Code'], errors='coerce')
    new_sales_df['Sales'] = new_sales_df['Quantity'] * new_sales_df['Product Amount']

    combined_df = pd.concat([hist_df, new_sales_df], ignore_index=True)
    combined_df['Warehouse ID'] = combined_df.apply( lambda row: get_reassigned_warehouse(row, pin_to_coord, fc_to_pincode), axis=1)
    combined_df['Warehouse ID'] = combined_df['Warehouse ID'].astype(str).str.strip()
    combined_df['Cluster'] = combined_df['Warehouse ID'].map(warehouse_to_cluster)
    combined_df['Cluster'].fillna('Unknown', inplace=True) 
    combined_df.drop_duplicates( subset=[ "Customer Shipment Date", "ASIN", "Quantity", "Product Amount", "Warehouse ID", "Cluster", "Ship Postal Code", "Sales"], keep="last", inplace=True)
    combined_df = combined_df.sort_values('Customer Shipment Date')
    write_df(TAB_COMBINED, combined_df)

    latest_date = combined_df['Customer Shipment Date'].max()
    prediction_start_date = latest_date + timedelta(days=1)
    prediction_end_date = prediction_start_date + timedelta(days=6)

    # ================== 9. MAJORITY/MINORITY SPLIT ==================
    result = combined_df.groupby(['ASIN', 'Cluster']).agg(Sales=('Sales', 'sum')).reset_index().sort_values(by='Sales', ascending=False)
    result['Category'] = 'Minority'
    result.loc[result['Sales'].cumsum() <= result['Sales'].sum() * 0.8, 'Category'] = 'Majority'
    majority_data_def = result[result['Category'] == 'Majority']
    minority_data_def = result[result['Category'] == 'Minority']
    write_df(TAB_MAJORITY_DATA, majority_data_def)
    write_df(TAB_MINORITY_DATA, minority_data_def)

    # Prediction Logic based on `time_unit`
    if time_unit == "Week":
        print("Running WEEKLY Majority forecast...")

        combined_df['Customer Shipment Date'] = pd.to_datetime( combined_df['Customer Shipment Date'], format='%Y-%m-%d', errors='coerce')
        latest_date = pd.to_datetime(latest_date, format='%Y-%m-%d', errors='coerce')

        maj_hist_df = combined_df.merge( majority_data_def[['ASIN', 'Cluster']], on=['ASIN', 'Cluster'], how='inner').copy()

        datetime_series = pd.to_datetime(maj_hist_df['Customer Shipment Date'], errors='coerce')
        maj_hist_df['Week_Start_Date'] = datetime_series.dt.to_period('W').apply(lambda r: r.start_time)
        weekly_sales = (
            maj_hist_df
            .groupby(['ASIN', 'Cluster', 'Week_Start_Date'])
            .agg(Quantity=('Quantity', 'sum'))
            .reset_index()
        )

        def create_features(df):
            datetime_col = pd.to_datetime(df['Week_Start_Date'], errors='coerce')
            df['week_of_year'] = datetime_col.dt.isocalendar().week.astype(int)
            df['month'] = datetime_col.dt.month
            df['lag_1_week'] = df.groupby(['ASIN', 'Cluster'])['Quantity'].shift(1).fillna(0)
            df['rolling_mean_4_weeks'] = (
                df.groupby(['ASIN', 'Cluster'])['Quantity']
                .shift(1)
                .rolling(window=4, min_periods=1)
                .mean()
                .fillna(0)
            )

            def get_trend(series):
                if len(series) < 2:
                    return 0
                return linregress(x=np.arange(len(series)), y=series).slope

            df['sales_trend_last_4_weeks'] = (
                df.groupby(['ASIN', 'Cluster'])['Quantity']
                .shift(1)
                .rolling(window=4)
                .apply(get_trend, raw=True)
                .fillna(0)
            )
            return df

        featured_sales = create_features(weekly_sales)

        X = pd.get_dummies(
            featured_sales.drop(['Quantity', 'Week_Start_Date'], axis=1),
            columns=['ASIN', 'Cluster'],
            drop_first=True
        )
        y = featured_sales['Quantity']
        y_log = np.log1p(y)
        X_np = X.to_numpy(dtype=float, copy=False)
        y_np = y_log.to_numpy(dtype=float, copy=False)
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        X = X.loc[:, ~X.columns.duplicated()]
        feature_names = X.columns.tolist()

        # Use numpy to avoid dtype quirks
        X_np = X.to_numpy(dtype=float, copy=False)
        y_np = y_log.astype(float).to_numpy(copy=False)
        dtrain = xgb.DMatrix(X_np, label=y_np)

        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 5,
            'eta': 0.08,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1
        }
        model = xgb.train(params, dtrain, num_boost_round=200, evals=[(dtrain, 'train')], verbose_eval=False)
        print("ML model training complete.")

        # 3) Future week anchor (keep as Timestamp)
        future_date = latest_date + timedelta(weeks=1)

        pred_df = majority_data_def[['ASIN', 'Cluster']].copy()
        pred_df['Week_Start_Date'] = future_date.to_period('W').start_time  
        pred_df['Quantity'] = 0

        last_week_data = weekly_sales[weekly_sales['Week_Start_Date'] == weekly_sales['Week_Start_Date'].max()]
        pred_df = pd.merge(
            pred_df,
            last_week_data[['ASIN', 'Cluster', 'Quantity']],
            on=['ASIN', 'Cluster'],
            how='left',
            suffixes=('', '_last_week')
        )
        pred_df.rename(columns={'Quantity_last_week': 'lag_1_week'}, inplace=True)
        pred_df['lag_1_week'].fillna(0, inplace=True)
        pred_df['rolling_mean_4_weeks'] = pred_df['lag_1_week']
        pred_df = create_features(pred_df)
        pred_X = pd.get_dummies(
            pred_df.drop(['Week_Start_Date', 'Quantity'], axis=1, errors='ignore'),
            columns=['ASIN', 'Cluster'],
            drop_first=True
        )
        # Make prediction features numeric, deduplicate
        pred_X = pred_X.apply(pd.to_numeric, errors='coerce').fillna(0)
        pred_X = pred_X.loc[:, ~pred_X.columns.duplicated()]

        # Strictly align to training features: add missing, drop extras, same order
        pred_X = pred_X.reindex(columns=feature_names, fill_value=0)

        # Use numpy for XGBoost
        dtest = xgb.DMatrix(pred_X.to_numpy(dtype=float, copy=False))

        log_predictions = model.predict(dtest)

        pred_df['predicted_demand'] = np.round(np.expm1(log_predictions * 1.6)).astype(int)
        pred_df['predicted_demand'] = pred_df['predicted_demand'].apply(lambda x: max(0, x))
        pred_df.insert(
            pred_df.columns.get_loc('predicted_demand') + 1,
            'predicted_demand_6_weeks',
            pred_df['predicted_demand'] * 6
        )

        last_order_info = (
            combined_df
            .groupby(['ASIN', 'Cluster'])['Customer Shipment Date'].max().reset_index(name='last_order_date')
        )
        # Ensure datetime for subtraction (combined_df already normalized; this is extra-safe)
        last_order_info['last_order_date'] = pd.to_datetime(
            last_order_info['last_order_date'], format='%Y-%m-%d', errors='coerce'
        )

        pred_df = pd.merge(pred_df, last_order_info, on=['ASIN', 'Cluster'], how='left')
        pred_df['days_since_last_order'] = (latest_date - pred_df['last_order_date']).dt.days
        pred_df['is_old_order'] = pred_df['days_since_last_order'] > 90

        mask = pred_df['days_since_last_order'] < 75
        maj_df = pred_df.loc[mask, ['ASIN', 'Cluster', 'predicted_demand', 'last_order_date', 'days_since_last_order', 'is_old_order']].copy()

        print("Running WEEKLY Minority forecast...")
        minority_sales = []

        minority_results = []
        for (asin, cluster), sales_hist in combined_df[
            combined_df.set_index(['ASIN', 'Cluster']).index.isin(
                minority_data_def.set_index(['ASIN', 'Cluster']).index
            )
        ].groupby(['ASIN', 'Cluster']):
            minority_sales.append(sales_hist)
            four_weeks_ago = latest_date - timedelta(weeks=4)
            last_4_weeks_sales = sales_hist[sales_hist['Customer Shipment Date'] > four_weeks_ago]
            total_quantity_last_4_weeks = last_4_weeks_sales['Quantity'].sum()
            predicted_demand = int(math.ceil(total_quantity_last_4_weeks / 4.0))

            last_order_date = sales_hist['Customer Shipment Date'].max()
            days_since_last_order = (latest_date - last_order_date).days if pd.notna(last_order_date) else np.nan

            if predicted_demand > 0 and days_since_last_order < 75:
                minority_results.append({
                    'ASIN': asin,
                    'Cluster': cluster,
                    'predicted_demand': predicted_demand,
                    'last_order_date': last_order_date,
                    'days_since_last_order': days_since_last_order
                    })

        min_df = pd.DataFrame(minority_results)

    elif time_unit == "Month":

        print("Running MONTHLY forecast...")
        majority_results = []
        latest_date_ts = pd.to_datetime(latest_date).date()
        window_size = 30
        num_windows = 12 

        def compute_ema(values, alpha=0.5):
            if not values:
                return 0.0
            ema = float(values[0])
            for v in values[1:]:
                ema = alpha * float(v) + (1 - alpha) * ema
            return ema

        for (asin, cluster), group in majority_data_def.groupby(['ASIN', 'Cluster']):
            sales_hist = combined_df[(combined_df['ASIN'] == asin) & (combined_df['Cluster'] == cluster)].copy()
            sales_hist['Customer Shipment Date'] = pd.to_datetime(sales_hist['Customer Shipment Date'], errors='coerce').dt.date
            if sales_hist.empty:
                continue

            window_sums = []
            window_starts = []
            for w in range(num_windows):
                end_date = latest_date_ts - timedelta(days=w * window_size)
                start_date = end_date - timedelta(days=window_size)
                mask = (sales_hist['Customer Shipment Date'] > start_date) & (sales_hist['Customer Shipment Date'] <= end_date)
                qty = sales_hist.loc[mask, 'Quantity'].sum()
                window_sums.append(int(qty))
                window_starts.append(start_date)

            window_sums = window_sums[::-1]
            window_starts = window_starts[::-1]

            last_month_sales = window_sums[-1]

            hist_vals = window_sums[:-1] 
            recent_hist = hist_vals[-6:] if len(hist_vals) >= 6 else hist_vals
            if len(recent_hist) > 0 and any(v != 0 for v in recent_hist):
                non_leading = recent_hist
                mean_prev_months = float(np.mean(non_leading))
            else:
                mean_prev_months = float(last_month_sales)

            tail = window_sums[-6:] if len(window_sums) >= 6 else window_sums
            if len(tail) >= 2:
                X = np.arange(len(tail)).reshape(-1, 1)
                y = np.array(tail, dtype=float)
                trend = LinearRegression().fit(X, y).coef_[0]
            else:
                trend = 0.0

            if len(window_sums) >= 12:
                seasonality = float(window_sums[-12])
            else:
                seasonality = float(mean_prev_months)
            ema = compute_ema(window_sums[-12:] if len(window_sums) >= 12 else window_sums, alpha=0.99)

            predicted_demand = (
                0.4 * float(last_month_sales) +
                0.8 * float(mean_prev_months) +
                0.8 * float(trend) +
                0.8 * float(seasonality) +
                0.5 * float(ema)
            )

            predicted_demand =  int(math.ceil(predicted_demand))
            last_order_date = sales_hist['Customer Shipment Date'].max()
            days_since_last_order = (latest_date_ts - last_order_date).days if pd.notna(last_order_date) else np.nan

            if predicted_demand > 0 and days_since_last_order < 50:
                majority_results.append({
                    'ASIN': asin,
                    'Cluster': cluster,
                    'predicted_demand': predicted_demand,
                    'last_order_date': last_order_date,
                    'days_since_last_order': days_since_last_order,
                })

        maj_df = pd.DataFrame(majority_results)
        maj_df['is_old_order'] = maj_df['days_since_last_order'] > 90
        maj_df = maj_df[['ASIN', 'Cluster', 'predicted_demand', 'last_order_date', 'days_since_last_order', 'is_old_order']]

        print("majority done")

        print("majority done")

        minority_results = []
        latest_date_ts = pd.to_datetime(latest_date).date()
        window_size = 30
        num_windows = 12

        for (asin, cluster), group in minority_data_def.groupby(['ASIN', 'Cluster']):
            sales_hist = combined_df[(combined_df['ASIN'] == asin) & (combined_df['Cluster'] == cluster)].copy()
            sales_hist['Customer Shipment Date'] = pd.to_datetime(sales_hist['Customer Shipment Date'], errors='coerce').dt.date
            if sales_hist.empty:
                continue
            last_order_date = sales_hist['Customer Shipment Date'].max()
            days_since_last_order = (latest_date_ts - last_order_date).days if pd.notna(last_order_date) else np.nan
            period_sales = []
            for w in range(num_windows):
                end_date = latest_date_ts - timedelta(days=w*window_size)
                start_date = end_date - timedelta(days=window_size)
                mask = (sales_hist['Customer Shipment Date'] >= start_date) & (sales_hist['Customer Shipment Date'] <= end_date)
                qty = sales_hist.loc[mask, 'Quantity'].sum()
                period_sales.append(qty)
            period_sales = period_sales[::-1]  
            if not period_sales:
                predicted_demand = 0
            else:
                weights = np.linspace(1, 3, num=len(period_sales))
                weighted_avg_sales = np.average(period_sales, weights=weights) if len(period_sales) > 1 else period_sales[-1]
                predicted_demand = max(4, int(round(weighted_avg_sales)))

            if predicted_demand > 0 and days_since_last_order < 50:
                minority_results.append({
                    'ASIN': asin,
                    'Cluster': cluster,
                    'predicted_demand': predicted_demand,
                    'last_order_date': last_order_date,
                    'days_since_last_order': days_since_last_order
                })

        min_df = pd.DataFrame(minority_results)
        min_df['is_old_order'] = min_df['days_since_last_order'] > 90
        min_df = min_df[['ASIN', 'Cluster', 'predicted_demand', 'last_order_date', 'days_since_last_order', 'is_old_order']]

    # --- Apply prediction choice scaling ---
    maj_df['predicted_demand'] = maj_df['predicted_demand'] * prediction_choice
    min_df['predicted_demand'] = min_df['predicted_demand'] * prediction_choice

    # --- Prepare Stock and Order Info ---
    inv_df.columns = inv_df.columns.str.strip()
    inv_df['Location'] = inv_df['Location'].astype(str).str.strip()
    inv_df['ASIN'] = inv_df['ASIN'].astype(str).str.strip()
    inv_df['Ending Warehouse Balance'] = pd.to_numeric(inv_df['Ending Warehouse Balance'], errors='coerce').fillna(0).astype(int)
    inv_df['Cluster'] = inv_df['Location'].map(warehouse_to_cluster)
    inv_df = inv_df[inv_df['Cluster'].notna()]
    inv_df['Date'] = pd.to_datetime(inv_df['Date'], errors='coerce')
    latest_dates = inv_df.groupby(['ASIN', 'Cluster'])['Date'].transform('max')
    inv_latest = inv_df[inv_df['Date'] == latest_dates]
    stock_df = inv_latest.groupby(['ASIN', 'Cluster'])['Ending Warehouse Balance'].sum().reset_index()
    stock_df.rename(columns={'Ending Warehouse Balance': 'stock_available'}, inplace=True)

    for df in [maj_df, min_df]:
        df['ASIN'] = df['ASIN'].astype(str).str.strip()
        df['Cluster'] = df['Cluster'].astype(str).str.strip()

    # --- Merge and Finalize Both DataFrames ---
    final_maj_df = maj_df.copy()
    final_min_df = min_df.copy()

    # Merge stock data
    final_maj_df = pd.merge(final_maj_df, stock_df, on=['ASIN', 'Cluster'], how='left')
    final_maj_df['stock_available'] = final_maj_df['stock_available'].fillna(0).astype(int)

    final_min_df = pd.merge(final_min_df, stock_df, on=['ASIN', 'Cluster'], how='left')
    final_min_df['stock_available'] = final_min_df['stock_available'].fillna(0).astype(int)

    # Calculate 'stock_to_send' as predicted demand - stock available
    final_maj_df['stock_to_send'] = final_maj_df['predicted_demand'] - final_maj_df['stock_available']
    final_maj_df['stock_to_send'] = final_maj_df['stock_to_send'].apply(lambda x: max(0, x))  # If negative, set to 0

    final_min_df['stock_to_send'] = final_min_df['predicted_demand'] - final_min_df['stock_available']
    final_min_df['stock_to_send'] = final_min_df['stock_to_send'].apply(lambda x: max(0, x))  # If negative, set to 0


    combined_df['ASIN'] = combined_df['ASIN'].astype(str).str.strip()
    combined_df['Cluster'] = combined_df['Cluster'].astype(str).str.strip()
    combined_df['Warehouse ID'] = combined_df['Warehouse ID'].astype(str).str.strip()

    warehouse_sales = combined_df.groupby(['ASIN', 'Cluster', 'Warehouse ID'])['Sales'].sum().reset_index()

    def get_top_3_from_group(group):
        sorted_group = group.sort_values(by='Sales', ascending=False)
        top_warehouses = sorted_group['Warehouse ID'].head(3).tolist()
        return ', '.join(top_warehouses)

    # 4. Apply the function to each ASIN-Cluster group to get the final list.
    top_3_warehouses = (
        warehouse_sales.groupby(['ASIN', 'Cluster'])
        .apply(get_top_3_from_group)
        .reset_index(name='Top_3_Warehouses')
    )
    
    write_df(TAB_TOP3, top_3_warehouses)

    final_maj_df = pd.merge(final_maj_df, top_3_warehouses, on=['ASIN', 'Cluster'], how='left')
    final_maj_df = final_maj_df[['ASIN', 'Cluster', 'Top_3_Warehouses'] + [col for col in final_maj_df.columns if col not in ['ASIN', 'Cluster', 'Top_3_Warehouses']]]

    final_min_df = pd.merge(final_min_df, top_3_warehouses, on=['ASIN', 'Cluster'], how='left')
    final_min_df = final_min_df[['ASIN', 'Cluster', 'Top_3_Warehouses'] + [col for col in final_min_df.columns if col not in ['ASIN', 'Cluster', 'Top_3_Warehouses']]]

    return final_maj_df, final_min_df, latest_date


# -------------- Example of how to use the updated function -----------------

# (Update these file paths as per your environment)
# sales_file_path = '/content/sample_data/66586020300.csv'
# cluster_wh_file_path = '/content/sample_data/Cluster_Warehouse_Mapping.xlsx'
# coord_cache_file = '/content/sample_data/postal_code_coords.csv'
# inv_file = '/content/sample_data/66587020300.csv'
# historical_data_path = '/content/sample_data/historical_sales_data_6jul_29jun.xlsx'
# classification_file_path = '/content/sample_data/static_asin_classification.csv'

# final_maj_df, final_min_df, latest_date = run_full_forecast(
#     sales_file_path=sales_file_path,
#     cluster_wh_file_path=cluster_wh_file_path,
#     coord_cache_file=coord_cache_file,
#     inv_file=inv_file,
#     historical_data_path=historical_data_path,
#     classification_file_path=classification_file_path, 
#     time_unit="Month", # Or "Week"
#     prediction_choice=1 # Or another integer
# )

# print("Majority DF Sample (using static classification):")
# print(final_maj_df.head())
# # final_maj_df.to_excel('/content/sample_data/maj_static.xlsx', index=False)
# print("\nMinority DF Sample (using static classification):")
# print(final_min_df.head())
# # final_min_df.to_excel('/content/sample_data/min_static.xlsx', index=False)