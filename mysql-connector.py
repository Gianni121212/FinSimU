import mysql.connector
import pandas as pd

# 讀取 CSV 檔案
df_sp = pd.read_csv("sp-100-index-03-14-2025.csv")
df_nasdaq = pd.read_csv("nasdaq-100-index-03-14-2025.csv")

# 只選取 Symbol 和 Name 欄位
df_sp = df_sp[['Symbol', 'Name']]
df_nasdaq = df_nasdaq[['Symbol', 'Name']]

# 合併兩個資料集，移除重複的 Symbol
df_all = pd.concat([df_sp, df_nasdaq], ignore_index=True).drop_duplicates(subset=['Symbol'])

# 移除 Symbol 或 Name 欄位為 NaN 的行
df_all = df_all.dropna(subset=['Symbol', 'Name'])

# 修正 Symbol 格式（去除空白並限制長度）
df_all['Symbol'] = df_all['Symbol'].str.strip().str[:50]  # 限制最大 50 字元

# 連接 MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="0912559910",
    database="testdb"
)
cursor = conn.cursor()

# 刪除舊的 stocks 資料表，重新建立
cursor.execute("DROP TABLE IF EXISTS stocks")
cursor.execute("""
CREATE TABLE stocks (
    symbol VARCHAR(50) PRIMARY KEY,
    name VARCHAR(255)
)
""")

# 插入新資料
insert_query = "INSERT INTO stocks (symbol, name) VALUES (%s, %s)"
for index, row in df_all.iterrows():
    cursor.execute(insert_query, (row['Symbol'], row['Name']))

# 提交變更
conn.commit()

# 查詢資料並顯示
cursor.execute("SELECT * FROM stocks")
rows = cursor.fetchall()
for row in rows:
    print(row)

# 關閉連線
cursor.close()
conn.close()

print("資料已成功匯入 MySQL 並更新 stocks 資料表。")
