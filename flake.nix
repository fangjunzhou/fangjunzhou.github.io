{
  description = "Fangjun's Hugo site";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    typo = {
      url = "github:tomfran/typo";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, flake-utils, typo }:
    flake-utils.lib.eachSystem [ "x86_64-linux" "x86_64-darwin" "aarch64-darwin" ]
      (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          themes = pkgs.linkFarm "hugo-themes" [{ name = "typo"; path = typo; }];
          preview = pkgs.writeShellScript "preview-site" ''
            exec ${pkgs.hugo}/bin/hugo server --themesDir "${themes}" \
              --poll 500ms --disableFastRender \
              --baseURL http://127.0.0.1/ --bind 127.0.0.1 "$@"
          '';
          browserExtension = pkgs.runCommand "blog-markdown-reference" {
            nativeBuildInputs = [ pkgs.zip ];
          } ''
            extension="$out/share/blog-markdown-reference"
            mkdir -p "$extension"
            cp ${./browser-extension}/{manifest.json,background.js,content.js} "$extension/"
            cd "$extension"
            TZ=UTC touch -t 198001010000 manifest.json background.js content.js
            TZ=UTC zip -qX "$out/blog-markdown-reference.zip" manifest.json background.js content.js
            cp "$out/blog-markdown-reference.zip" "$out/blog-markdown-reference.xpi"
          '';
        in {
          packages.default = pkgs.stdenvNoCC.mkDerivation {
            name = "fangjunzhou.github.io";
            src = self;
            nativeBuildInputs = [ pkgs.hugo ];
            buildPhase = ''
              hugo --themesDir "${themes}" --cacheDir "$TMPDIR/hugo-cache" --minify --destination "$out"
            '';
            dontInstall = true;
          };
          packages.browser-extension = browserExtension;

          apps.default = {
            type = "app";
            program = "${preview}";
            meta.description = "Preview the site from the current checkout";
          };

          devShells.default = pkgs.mkShell {
            packages = [
              pkgs.hugo
              pkgs.git
              pkgs.uv
              pkgs.python313
              pkgs.pkg-config
              pkgs.cairo
              pkgs.pango
              (pkgs.texlive.withPackages (tex: [
                tex.scheme-basic
                tex.standalone
                tex.preview
                tex.dvisvgm
              ]))
            ];
            HUGO_THEMESDIR = themes;
          };
        });
}
