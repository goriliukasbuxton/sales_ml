
import pandas as pd

# ⬇ Point this to where you downloaded your CSV:
local_csv = r'C:\Users\laury\Desktop\corteva\ds_ml_task\GRIDMET_Drought_Monthly_By_State.csv'
df = pd.read_csv(local_csv)

# 1) Keep only state, date, pdsi
df = df[['state', 'date', 'pdsi']].copy()

# 2) Extract “year” and “month” from “date” (which is in YYYY-MM format)
df['year']  = df['date'].str.slice(0, 4)
df['month'] = df['date'].str.slice(5, 7)   # e.g. "01", "02", …, "12"

# 3) Pivot into wide form: index = [state, year], columns = month, values = pdsi
pivot = df.pivot(index=['state', 'year'], columns='month', values='pdsi')

# 4) Rename each column to append “_PDSI” (so “01” → “01_PDSI”, etc.)
pivot.columns = [f"{m}_PDSI" for m in pivot.columns]

# 5) Reset index so “state” and “year” become ordinary columns again
pivot = pivot.reset_index()

# 6) Inspect the first few rows
print(pivot.head(10))

# # 8) (Optionally) write out the wide table to disk:
pivot.to_csv('GRIDMET_PDSI_State_Year_Wide_2021_2025.csv', index=False)

