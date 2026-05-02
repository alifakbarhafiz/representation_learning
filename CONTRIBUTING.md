## Contributing

Thanks for your interest in contributing.

### Development setup

- **Python**: 3.8+
- **Install**:

```bash
python -m venv .venv
# Windows PowerShell:
# .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt
```

### Running tests

From the `project/` directory:

```bash
pytest tests/ -v
```

### Style and scope

- Keep changes **modular** (edit the relevant module under `data/`, `models/`, `representations/`, `utils/`).
- Prefer adding/adjusting **tests** alongside code changes.
- Avoid committing large artifacts (datasets, checkpoints, generated plots). The repository ignores common artifact folders via `.gitignore`.

