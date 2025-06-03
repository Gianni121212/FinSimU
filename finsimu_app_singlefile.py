import os
import sqlite3
import bcrypt # For password hashing
import yfinance as yf
from flask import Flask, request, jsonify, session # Using session for simplicity
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai # For Gemini

# --- Configuration and Initialization ---
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')
CORS(app) # Allow all origins for simplicity in development

DATABASE_FILE = os.environ.get('DATABASE_FILE', 'finsimu_single_file.db')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-pro') # Or your preferred model
        print("Gemini API configured successfully.")
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")
        gemini_model = None
else:
    print("GEMINI_API_KEY not found in .env file. Gemini features will be disabled.")
    gemini_model = None

# --- Database Helper Functions (SQLite) ---
def get_db_connection():
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row # Access columns by name
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    # Users Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            risk_propensity TEXT,
            cash_balance REAL DEFAULT 100000.00,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # Trades Table (Simplified)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            ticker TEXT NOT NULL,
            type TEXT NOT NULL, -- 'buy' or 'sell'
            shares INTEGER NOT NULL,
            price_at_trade REAL NOT NULL,
            mood TEXT,
            reason TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    # Portfolio Holdings (Simplified)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS portfolio_holdings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            ticker TEXT NOT NULL,
            shares INTEGER NOT NULL,
            average_cost REAL NOT NULL,
            UNIQUE (user_id, ticker), -- Ensure one entry per user per stock
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    conn.commit()
    conn.close()
    print("Database initialized.")

# Initialize DB when app starts
init_db()

# --- API Routes ---

@app.route('/api/hello', methods=['GET'])
def hello():
    return jsonify({"message": "Hello from FinSimU Single-File API!"})

# --- User Authentication (Simplified with Session) ---
@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    risk_propensity = data.get('risk_propensity', 'conservative')
    start_capital = float(data.get('startCapital', 100000.00))

    if not all([username, email, password]):
        return jsonify({"error": "Missing required fields"}), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        if cursor.fetchone():
            return jsonify({"error": "Username already exists"}), 409
        cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
        if cursor.fetchone():
            return jsonify({"error": "Email already exists"}), 409

        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        cursor.execute(
            "INSERT INTO users (username, email, password_hash, risk_propensity, cash_balance) VALUES (?, ?, ?, ?, ?)",
            (username, email, hashed_password, risk_propensity, start_capital)
        )
        conn.commit()
        return jsonify({"message": "User registered successfully. Please login."}), 201
    except sqlite3.Error as e:
        return jsonify({"error": f"Database error: {e}"}), 500
    finally:
        conn.close()

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()

    if user and bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
        session['user_id'] = user['id'] # Store user_id in session
        session['username'] = user['username']
        return jsonify({
            "message": "Login successful",
            "user": {
                "id": user['id'],
                "username": user['username'],
                "email": user['email'],
                "risk_propensity": user['risk_propensity'],
                "cash_balance": user['cash_balance']
            }
        }), 200
    else:
        return jsonify({"error": "Invalid username or password"}), 401

@app.route('/api/auth/logout', methods=['POST'])
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    return jsonify({"message": "Logout successful"}), 200

@app.route('/api/user/profile', methods=['GET'])
def get_user_profile():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    user_id = session['user_id']
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, email, risk_propensity, cash_balance FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    conn.close()

    if user:
        return jsonify(dict(user)), 200
    else:
        return jsonify({"error": "User not found"}), 404


# --- Stock Data (yfinance) ---
@app.route('/api/stocks/quote/<ticker>', methods=['GET'])
def stock_quote(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        # print(f"Fetched info for {ticker}: {info.get('regularMarketPrice')}") # Debugging
        if not info or 'regularMarketPrice' not in info or info.get('regularMarketPrice') is None:
             # Try history if info is incomplete (common for some tickers/markets)
            hist = stock.history(period="1d")
            if not hist.empty:
                latest_price = hist['Close'].iloc[-1]
                # For change, we might need more data or a different approach
                return jsonify({
                    "symbol": ticker.upper(),
                    "shortName": info.get('shortName', ticker.upper()), # Fallback to ticker
                    "regularMarketPrice": float(latest_price) if latest_price else 0.0,
                    "regularMarketChangePercent": 0.0 # Placeholder for simplicity
                })
            else:
                 return jsonify({"error": f"Could not retrieve price for {ticker}"}), 404


        return jsonify({
            "symbol": info.get('symbol'),
            "shortName": info.get('shortName'),
            "regularMarketPrice": info.get('regularMarketPrice'),
            "regularMarketChangePercent": info.get('regularMarketChangePercent', 0.0) * 100 if info.get('regularMarketChangePercent') else 0.0, # yfinance gives it as decimal
            # Add more fields if your frontend needs them
        })
    except Exception as e:
        print(f"yfinance error for {ticker}: {e}")
        return jsonify({"error": f"Failed to fetch data for {ticker}: {e}"}), 500

# --- Trading Logic (Simplified) ---
@app.route('/api/trades', methods=['POST'])
def execute_trade():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized, please login"}), 401

    user_id = session['user_id']
    data = request.get_json()
    ticker = data.get('ticker')
    trade_type = data.get('type') # 'buy' or 'sell'
    shares = int(data.get('shares', 0))
    price_at_trade = float(data.get('price_at_trade', 0)) # Price from frontend at time of trade
    mood = data.get('mood')
    reason = data.get('reason')

    if not all([ticker, trade_type, shares > 0, price_at_trade > 0]):
        return jsonify({"error": "Missing or invalid trade data"}), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Get user's cash balance
        cursor.execute("SELECT cash_balance FROM users WHERE id = ?", (user_id,))
        user_row = cursor.fetchone()
        if not user_row:
            return jsonify({"error": "User not found"}), 404
        cash_balance = user_row['cash_balance']

        total_value = shares * price_at_trade

        if trade_type == 'buy':
            if cash_balance < total_value:
                return jsonify({"error": "Insufficient cash balance"}), 400
            new_cash_balance = cash_balance - total_value
            # Update portfolio
            cursor.execute("SELECT shares, average_cost FROM portfolio_holdings WHERE user_id = ? AND ticker = ?", (user_id, ticker))
            holding = cursor.fetchone()
            if holding:
                new_total_shares = holding['shares'] + shares
                new_avg_cost = ((holding['average_cost'] * holding['shares']) + total_value) / new_total_shares
                cursor.execute(
                    "UPDATE portfolio_holdings SET shares = ?, average_cost = ? WHERE user_id = ? AND ticker = ?",
                    (new_total_shares, new_avg_cost, user_id, ticker)
                )
            else:
                cursor.execute(
                    "INSERT INTO portfolio_holdings (user_id, ticker, shares, average_cost) VALUES (?, ?, ?, ?)",
                    (user_id, ticker, shares, price_at_trade)
                )
        elif trade_type == 'sell':
            cursor.execute("SELECT shares FROM portfolio_holdings WHERE user_id = ? AND ticker = ?", (user_id, ticker))
            holding = cursor.fetchone()
            if not holding or holding['shares'] < shares:
                return jsonify({"error": "Insufficient shares to sell"}), 400
            new_cash_balance = cash_balance + total_value
            new_total_shares = holding['shares'] - shares
            if new_total_shares == 0:
                cursor.execute("DELETE FROM portfolio_holdings WHERE user_id = ? AND ticker = ?", (user_id, ticker))
            else:
                cursor.execute("UPDATE portfolio_holdings SET shares = ? WHERE user_id = ? AND ticker = ?", (new_total_shares, user_id, ticker))
        else:
            return jsonify({"error": "Invalid trade type"}), 400

        # Update user's cash balance
        cursor.execute("UPDATE users SET cash_balance = ? WHERE id = ?", (new_cash_balance, user_id))
        # Record the trade
        cursor.execute(
            "INSERT INTO trades (user_id, ticker, type, shares, price_at_trade, mood, reason) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, ticker, trade_type, shares, price_at_trade, mood, reason)
        )
        conn.commit()
        return jsonify({
            "message": f"Trade executed successfully: {trade_type.capitalize()} {shares} of {ticker}",
            "new_cash_balance": new_cash_balance
        }), 200
    except sqlite3.Error as e:
        conn.rollback()
        return jsonify({"error": f"Database error during trade: {e}"}), 500
    finally:
        conn.close()

# --- Portfolio (Simplified) ---
@app.route('/api/portfolio', methods=['GET'])
def get_portfolio():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    user_id = session['user_id']
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Get cash balance
        cursor.execute("SELECT cash_balance FROM users WHERE id = ?", (user_id,))
        user_data = cursor.fetchone()
        cash_balance = user_data['cash_balance'] if user_data else 0

        # Get holdings
        cursor.execute("SELECT ticker, shares, average_cost FROM portfolio_holdings WHERE user_id = ?", (user_id,))
        holdings_raw = cursor.fetchall()
        holdings_processed = []
        total_portfolio_value = cash_balance # Start with cash

        for row in holdings_raw:
            ticker_data = yf.Ticker(row['ticker']).info
            current_price = ticker_data.get('regularMarketPrice')
            # Fallback for current price if regularMarketPrice is missing
            if current_price is None:
                hist = yf.Ticker(row['ticker']).history(period="1d")
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                else:
                    current_price = row['average_cost'] # Fallback to avg_cost if no price found

            market_value = row['shares'] * current_price
            total_pl = (current_price - row['average_cost']) * row['shares']
            total_pl_percent = ((current_price / row['average_cost']) - 1) * 100 if row['average_cost'] > 0 else 0

            holdings_processed.append({
                "ticker": row['ticker'],
                "name": ticker_data.get('shortName', row['ticker']),
                "shares": row['shares'],
                "average_cost": row['average_cost'],
                "current_price": current_price,
                "market_value": market_value,
                "total_pl": total_pl,
                "total_pl_percent": total_pl_percent,
                "today_change_percent": (ticker_data.get('regularMarketChangePercent', 0) or 0) * 100, # Ensure it's a number
            })
            total_portfolio_value += market_value

        # Calculate total P/L based on initial total capital (e.g., 100k if not tracked differently)
        # For a more accurate total P/L, you'd sum P/L of all holdings and compare current total_portfolio_value to initial capital + net deposits/withdrawals
        # Here, a simpler version:
        initial_total_capital = 100000.00 # Assuming this for now, or fetch user's initial capital
        overall_total_pl_value = total_portfolio_value - initial_total_capital # This isn't perfect without tracking all cash movements
        overall_total_pl_percent = (overall_total_pl_value / initial_total_capital) * 100 if initial_total_capital > 0 else 0


        return jsonify({
            "overview": {
                "portfolio_value": total_portfolio_value,
                "cash_balance": cash_balance,
                "total_pl_value": sum(h['total_pl'] for h in holdings_processed), # Sum of P/L from individual holdings
                "total_pl_percent": overall_total_pl_percent, # This is an approximation
                "today_pl_value": 0.00, # Placeholder, requires more complex calculation
                "today_pl_percent": 0.00 # Placeholder
            },
            "holdings": holdings_processed
        })
    except Exception as e:
        print(f"Portfolio fetch error: {e}")
        return jsonify({"error": f"Error fetching portfolio: {str(e)}"}), 500
    finally:
        conn.close()

# --- AI Reports (Gemini - Simplified Placeholder) ---
@app.route('/api/ai/investment_report', methods=['POST'])
def ai_investment_report():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    if not gemini_model:
        return jsonify({"error": "Gemini API not configured or unavailable"}), 503

    user_id = session['user_id']
    # In a real app, fetch user's trade history, portfolio, mood logs
    # For now, use a placeholder prompt
    prompt = f"Generate a concise investment performance report and behavioral insights for user {user_id} based on their simulated trading. User has made 5 trades, with 3 profitable. They tend to trade tech stocks and selected 'confident' mood often. Provide 2 actionable suggestions."
    try:
        response = gemini_model.generate_content(prompt)
        return jsonify({"report": response.text})
    except Exception as e:
        print(f"Gemini API error for investment report: {e}")
        return jsonify({"error": f"Failed to generate AI report: {str(e)}"}), 500

@app.route('/api/ai/behavioral_analysis', methods=['POST'])
def ai_behavioral_analysis():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    if not gemini_model:
        return jsonify({"error": "Gemini API not configured or unavailable"}), 503

    user_id = session['user_id']
    prompt = f"Analyze potential behavioral biases for user {user_id} who often feels 'anxious' before selling losing stocks and 'excited' when buying on market dips. Provide one key observation and one tip."
    try:
        response = gemini_model.generate_content(prompt)
        return jsonify({"analysis": response.text})
    except Exception as e:
        print(f"Gemini API error for behavioral analysis: {e}")
        return jsonify({"error": f"Failed to generate AI analysis: {str(e)}"}), 500


# --- Main Execution ---
if __name__ == '__main__':
    # Make sure to create .env file with SECRET_KEY and GEMINI_API_KEY
    # Example: FLASK_APP=finsimu_app_singlefile.py flask run
    # Or: python finsimu_app_singlefile.py
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)