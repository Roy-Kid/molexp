# MolExp documentation layout

MolExp keeps one Zensical source tree per canonical language:

- `en/` is the English documentation source and is built by `zensical.toml`.
- `zh/` is the Simplified Chinese documentation source and is built by
  `zensical.zh.toml`.

This follows Zensical's language model: each generated site has one
`project.theme.language`, and the header language selector is configured through
`project.extra.alternate`.

Build both language sites from the repository root:

```bash
zensical build --strict
zensical build --strict -f zensical.zh.toml
```

The two builds write to `site/en` and `site/zh`. The Chinese tree may use
symlinks into `docs/en` for pages that have not been translated yet; replace a
symlink with a real Markdown file when translating that page.
