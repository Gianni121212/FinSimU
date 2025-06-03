# FinSimU - 虛擬理財訓練平台

FinSimU 是一款基於 Python + Flask 的虛擬理財系統，支援 GA 技術指標優化、股票模擬操作、圖形化回測與個股策略分析。

## 🚀 快速啟動

```bash
git clone https://github.com/Gianni121212/FinSimU.git
cd FinSimU
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
python app.py
```

## 📁 專案結構

- `app.py`：主後端應用
- `templates/`：HTML 前端模板
- `static/`：前端資源（icons、manifest 等）
- `ga_engine.py`：遺傳演算法核心
- `offline_ga_trainer.py`：離線優化腳本
- `requirements.txt`：套件清單
- `.env.example`：環境變數範本

## 🔐 環境變數

請依照 `.env.example` 建立 `.env` 檔案，並填入：

- `DATABASE_URL`：資料庫連線字串
- `SECRET_KEY`：Flask session 密鑰
- `GENAI_API_KEY`：Google Generative AI 金鑰（若使用 AI 分析）

## 🧠 使用技術

- Flask + Jinja2 模板引擎
- pandas / yfinance 數據處理
- plotly 可視化
- GA 遺傳演算法優化技術指標
