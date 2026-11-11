# cleanroom

A small, domain-neutral toolkit for honest evaluation of models on time-ordered data. It is built inside the [Tells](https://github.com/BK5102/tells) repo and shared with the sibling projects Reserve and Culprit.

Install a tagged release from another project:

```
uv add "cleanroom @ git+https://github.com/BK5102/tells@cleanroom-v0.1#subdirectory=packages/cleanroom"
```

Modules (built over Tells Days 1 to 11): `split` (time-ordered splits with a gap), `lock` (sealed-envelope config hash), `metrics`, `calibration`, `bootstrap`, `stats` (multiple-comparison correction), `canary` (planted-leak test), `provenance`.

Rules: no domain-specific code or vocabulary, no imports from any project package, and every change other projects need ships as a new `cleanroom-vX.Y` tag. See `CHANGELOG.md`.
