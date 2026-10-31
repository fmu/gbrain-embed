{
  description = "On-demand Qwen3 embeddings for gbrain, served by llama.cpp";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs =
    { self, nixpkgs }:
    let
      systems = [
        "aarch64-darwin"
        "x86_64-darwin"
        "aarch64-linux"
        "x86_64-linux"
      ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      packages = forAllSystems (pkgs: rec {
        gbrain-embed = pkgs.writeShellApplication {
          name = "gbrain-embed";
          runtimeInputs = [
            pkgs.bun
            pkgs.curl
            pkgs.coreutils
          ];
          text = ''
            exec ${pkgs.bun}/bin/bun ${./src}/cli.ts "$@"
          '';
        };
        default = gbrain-embed;
      });
    };
}
