# Fangjun Zhou

Fangjun Zhou's personal homepage, built with Hugo and the [Typo theme](https://github.com/tomfran/typo). From the repository root, use Nix with flakes enabled.

| Command | Purpose |
| --- | --- |
| `nix build` | Build the production site in `result/`. |
| `nix run` | Preview the working tree at <http://127.0.0.1:1313/> with live reload. |
| `nix develop` | Start a shell with Hugo, Git, and the pinned theme available. |

If direnv is hooked into your shell, run `direnv allow` once from the repository root. Entering the directory then loads the Nix development shell, syncs the locked uv workspace using the flake's Python, and activates `.venv`.

Leave `nix run` running while you edit. Saving `content/_index.md` rebuilds the site and refreshes the open homepage automatically.

To preview on a different port, run `nix run . -- --port 8000`. Add new site files to Git before running the production build so Nix includes them.

Create a draft with `nix develop -c hugo new content posts/my-first-post.md`. Preview drafts with `nix run . -- --buildDrafts`. Set `draft = false` when the post is ready to appear on the homepage and at `/posts/`.

## Copy a Markdown reference

Run `nix build .#browser-extension` to build the extension. Use `nix run` separately to start the blog preview.

- **Firefox** Open `about:debugging#/runtime/this-firefox` and use **Load Temporary Add-on** to open `result/blog-markdown-reference.xpi`. Permanent installation in standard Firefox requires Mozilla signing.
- **Chrome** Open `chrome://extensions`, enable **Developer mode**, then select `result/share/blog-markdown-reference/` with **Load unpacked**. The ZIP contains the same unpacked files. Regular installation on macOS requires Chrome Web Store distribution or an enterprise policy.

Reload any open posts after loading the extension. The live blog gains source references when the Hugo changes are deployed.

Right-click selected text or a post image or video and choose **Copy Markdown reference**. Text selections copy the repository-relative Markdown path and line range, followed by the selection on a new line. Images and videos copy the path and line range.

To use a keyboard shortcut, assign a key combination to **Copy Markdown reference**. In Firefox, open `about:addons`, then choose **Manage Extension Shortcuts** from the gear menu. In Chrome, open `chrome://extensions/shortcuts`. Use the shortcut with text selected or while hovering over an image or video.

## Deployment

In the repository’s **Settings → Pages**, select **GitHub Actions** as the build source. Each push to `main` then builds the site with Nix and publishes it to <https://fangjunzhou.github.io/>. To redeploy the current version, run the **Deploy homepage** workflow on `main` from the Actions tab.
