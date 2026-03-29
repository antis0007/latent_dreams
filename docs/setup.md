# Setup

## Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e .[llm,dev]
```

## Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e .
pip install -e .[llm,dev]
```

If wheel build fails for `llama-cpp-python`, install Visual Studio Build Tools and CMake.
