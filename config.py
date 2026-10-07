import os
from dotenv import load_dotenv

load_dotenv()   # opzionale: in locale puoi usare un .env; nel Codespace arrivano dai secrets di GitHub

try:
    API_KEY = os.environ["ALPACA_API_KEY"]
    API_SECRET = os.environ["ALPACA_API_SECRET"]
except KeyError as e:
    raise SystemExit(f"Manca la variabile {e}. Impostala come Codespaces secret su GitHub "
                     "(Settings > Codespaces) e riavvia il codespace.") from None
