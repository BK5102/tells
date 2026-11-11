# Changelog

## Unreleased (0.1.0.dev0)

- Package skeleton inside the Tells uv workspace.
- `cleanroom.split`: `Window`, `Split` (train, optional valid, test, `min_gap`), `apply`, `check`, `to_dict` / `from_dict`. Errors: `SplitError` (bad config), `LeakError` (data breaks time order or gap).
