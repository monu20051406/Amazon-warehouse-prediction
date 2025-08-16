import pandas as pd
from datetime import datetime
from gsheets_io import read_df, write_df, upsert_df  

TAB_INV_SUMMARY = "inventory_summary"

def inventory_ledger(inven_file, creds):

    if inven_file is not None:
        df = pd.read_csv(inven_file)
        df = df[['Date', 'ASIN', 'Ending Warehouse Balance', 'Location']]
        df = df.rename(columns={
            'Ending Warehouse Balance': 'EndingBalance',
            'Location': 'Warehouse'
        })

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.sort_values(['ASIN', 'Warehouse', 'Date'])

        summary_list = []
        for (asin, wh), group in df.groupby(['ASIN', 'Warehouse']):
            group_sorted = group.sort_values('Date')
            last_row = group_sorted.iloc[-1]
            current_stock = last_row['EndingBalance']

            # Detect last order date where stock decreased
            deltas = group_sorted['EndingBalance'].diff()
            last_order_date = group_sorted.loc[deltas < 0, 'Date'].max()

            if pd.notna(last_order_date):
                # If there is a valid last order date
                last_order_date_str = last_order_date.strftime('%Y-%m-%d')
                days_since_last_order = (datetime.now() - pd.to_datetime(last_order_date)).days
            else:
                # If no stock decrease (no valid last order date), set appropriate default values
                last_order_date_str = ""
                days_since_last_order = None  # or set it to a large value like 9999 if desired

            summary_list.append({
                "ASIN": asin,
                "Warehouse": wh,
                "Current Stock": current_stock,
                "Last Order Date": last_order_date_str,
                "Days Since Last Order": days_since_last_order
            })

        new_summary = pd.DataFrame(summary_list)
    else:
        new_summary = None

    try:
        master_df = read_df(TAB_INV_SUMMARY)
        keep_cols = ["ASIN", "Warehouse", "Current Stock", "Last Order Date", "Days Since Last Order"]
        master_df = master_df[[c for c in keep_cols if c in master_df.columns]]
    except Exception:
        master_df = pd.DataFrame(columns=[
            "ASIN", "Warehouse", "Current Stock", "Last Order Date", "Days Since Last Order"
        ])

    if new_summary is not None and not new_summary.empty:
        combined = pd.concat([master_df, new_summary], ignore_index=True)
        combined['Last Order Date'] = pd.to_datetime(combined['Last Order Date'], errors='coerce')
        combined = combined.sort_values('Last Order Date').drop_duplicates(['ASIN', 'Warehouse'], keep='last')
        combined['Last Order Date'] = combined['Last Order Date'].dt.strftime('%Y-%m-%d').fillna("")

        final_df = combined[combined['Current Stock'] != 0].reset_index(drop=True)

        write_df(TAB_INV_SUMMARY, final_df)

        return final_df
    
    return master_df
