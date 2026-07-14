← [Documentation Home](README.md)

---

# Configuration

### Overriding Defaults (config_user.py)

Settings in [`config.py`](../config.py) can be overridden by creating a `config_user.py` file in the same directory. This lets you persist custom settings across EasyLLM updates without modifying the main config — your overrides survive reinstallation and upgrades.

```python
# config_user.py — example overrides
REATTACH_IMAGES = True          # re-attach images across chat turns
HISTORY_DB_MAX_AGE_DAYS = 30   # auto-delete history older than 30 days
HISTORY_DB_MAX_SIZE_MB = 200   # cap history DB at 200 MB
```

Only the variables you explicitly set will be overridden; everything else keeps its default from `config.py`. A malformed `config_user.py` will log a warning and fall back to defaults — it won't crash ComfyUI.

See [`config.py`](../config.py) for the full list of configurable settings and their default values.

---

← [Back to Documentation](README.md)
