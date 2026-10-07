#!/usr/bin/env bash
# Setup del bot paper su una VM Ubuntu (utente ubuntu). Idempotente: si può rilanciare.
set -euo pipefail
sudo apt-get update -q && sudo apt-get install -y -q python3-venv git
[ -d ~/algo-lab ] || git clone https://github.com/vittorioiacovoni0-ux/algo-lab.git ~/algo-lab
cd ~/algo-lab && git pull -q
python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt
.venv/bin/pytest -q
sudo cp deploy/algo-pairs.service /etc/systemd/system/
sudo systemctl daemon-reload
if [ -f .env ]; then
  chmod 600 .env
  echo "Pronto. Avvia il bot con: sudo systemctl enable --now algo-pairs"
  echo "(prima fermalo sul PC: due istanze sullo stesso conto raddoppiano gli ordini)"
else
  echo "Manca ~/algo-lab/.env con ALPACA_API_KEY e ALPACA_API_SECRET (chiavi PAPER): crealo, poi rilancia."
fi
