# Fangjun Zhou

Fangjun Zhou's personal homepage, built with Hugo and the [Typo theme](https://github.com/tomfran/typo). From the repository root, use Nix with flakes enabled.

| Command | Purpose |
| --- | --- |
| `nix build` | Build the production site in `result/`. |
| `nix run` | Preview the working tree at <http://127.0.0.1:1313/> with live reload. |
| `nix develop` | Start a shell with Hugo, Git, and the pinned theme available. |

Leave `nix run` running while you edit. Saving `content/_index.md` rebuilds the site and refreshes the open homepage automatically.

To preview on a different port, run `nix run . -- --port 8000`. Add new site files to Git before running the production build so Nix includes them.

Create a draft with `nix develop -c hugo new content posts/my-first-post.md`. Preview drafts with `nix run . -- --buildDrafts`. Set `draft = false` when the post is ready to appear on the homepage and at `/posts/`.
