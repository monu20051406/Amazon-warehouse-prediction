import pandas as pd
import numpy as np
import gspread
from gspread_dataframe import get_as_dataframe, set_with_dataframe
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials

# Use Streamlit secrets for authentication
import streamlit as st

# Use the credentials stored in secrets.toml
SECRETS = st.secrets["gcp_service_account"]

# Construct credentials object from the secrets
creds = Credentials.from_service_account_info(
    SECRETS,
    scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
)

SPREADSHEET_ID = "1kQticbIs3s6HXkAvUaQ_Uf-9ZIooXQI-SkTTAyI4SVI"  
HEADER_ROW = 1 

_gc = None
def _client():
    global _gc
    if _gc is None:
        _gc = gspread.authorize(creds)  # Use credentials for authorization
    return _gc

def _open_sheet():
    return _client().open_by_key(SPREADSHEET_ID)

def _get_ws(sheet_title: str, create: bool = False, rows: int = 1000, cols: int = 26):
    sh = _open_sheet()
    try:
        return sh.worksheet(sheet_title)
    except gspread.exceptions.WorksheetNotFound:
        if not create:
            raise
        return sh.add_worksheet(title=sheet_title, rows=rows, cols=cols)

def read_df(sheet_title: str) -> pd.DataFrame:
    """Read a whole tab into a DataFrame (auto-strips empty rows/cols)."""
    ws = _get_ws(sheet_title, create=False)
    df = get_as_dataframe(ws, evaluate_formulas=True, header=HEADER_ROW-1)  # 0-index in API
    if len(df.columns):
        df.columns = [str(c).strip() for c in df.columns]
    # drop fully-empty rows/cols Google often leaves
    df = df.dropna(how="all").dropna(axis=1, how="all")
    return df

def write_df(sheet_title: str, df: pd.DataFrame):
    """Overwrite a tab with DataFrame (resizes sheet)."""
    ws = _get_ws(sheet_title, create=True, rows=max(2, len(df)+10), cols=max(1, len(df.columns)+5))
    sh = _open_sheet()
    try:
        old = sh.worksheet(sheet_title)
        sh.del_worksheet(old)
    except gspread.exceptions.WorksheetNotFound:
        pass
    ws = _get_ws(sheet_title, create=True, rows=max(2, len(df)+10), cols=max(1, len(df.columns)+5))
    set_with_dataframe(ws, df, include_index=False, include_column_header=True, resize=True)

def append_df(sheet_title: str, df_new: pd.DataFrame):
    """Append rows under the header. Assumes existing header matches df_new columns."""
    ws = _get_ws(sheet_title, create=True)
    current = get_as_dataframe(ws, evaluate_formulas=False, header=HEADER_ROW-1)
    if current.dropna(how="all").empty:
        write_df(sheet_title, df_new)
        return
    existing_cols = [c for c in current.columns if str(c) != "nan"]
    df_new = df_new.reindex(columns=existing_cols, fill_value=None)
    values = ws.get_all_values()
    next_row = len(values) + 1  # 1-based
    set_with_dataframe(ws, df_new, row=next_row, include_index=False, include_column_header=False, resize=False)

def upsert_df(sheet_title: str, df_new: pd.DataFrame, key_cols: list):
    """
    Overwrite existing rows that match on key_cols, append the rest.
    Requires that the sheet already has a header row.
    """
    try:
        df_old = read_df(sheet_title)
        if df_old.empty:
            write_df(sheet_title, df_new)
            return
    except gspread.exceptions.WorksheetNotFound:
        write_df(sheet_title, df_new)
        return

    all_cols = list(dict.fromkeys(list(df_old.columns) + list(df_new.columns)))
    df_old = df_old.reindex(columns=all_cols)
    df_new = df_new.reindex(columns=all_cols)

    def key_tuple(df):
        return tuple(df[k].astype(str).fillna("") for k in key_cols)

    old_index = pd.MultiIndex.from_arrays(key_tuple(df_old), names=key_cols)
    new_index = pd.MultiIndex.from_arrays(key_tuple(df_new), names=key_cols)

    common = old_index.intersection(new_index)
    if len(common):
        mask = ~old_index.isin(common)
        df_old_kept = df_old.loc[mask].copy()
        df_replacements = df_new.loc[new_index.isin(common)].copy()
        df_new_only = df_new.loc[~new_index.isin(common)].copy()
        result = pd.concat([df_old_kept, df_replacements, df_new_only], ignore_index=True)
    else:
        result = pd.concat([df_old, df_new], ignore_index=True)

    write_df(sheet_title, result)
