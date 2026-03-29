python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
pip install -e .[llm,dev]
Write-Host "GGUF Dream Lab environment ready."
