import google.generativeai as genai
from flask import Flask, render_template, request, jsonify, session, redirect
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
import warnings
import os
import secrets
import datetime as dt
import feedparser
import urllib.parse
from transformers import pipeline
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# 列出可用模型
models = genai.list_models()
for model in models:
    print(f"模型名稱: {model.name}, 支持的方法: {model.supported_generation_methods}")