# Fangjun's Dev Blog

A Hugo site using the [Typo theme](https://github.com/tomfran/typo). From the repository root, use Nix with flakes enabled.

| Command | Purpose |
| --- | --- |
| `nix build` | Build the production site in `result/`. |
| `nix run` | Preview the working tree at <http://127.0.0.1:1313/> with live reload. |
| `nix develop` | Start a shell with Hugo, Git, and the pinned theme available. |

To preview on a different port, run `nix run -- --port 8000`. Add new site files to Git before running the production build so Nix includes them.
