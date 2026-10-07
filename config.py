import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ["ALPACA_API_KEY"]
API_SECRET = os.environ["ALPACA_API_SECRET"]
