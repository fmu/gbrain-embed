{
  description = "On-demand Qwen3 embeddings and reranking for gbrain, served by llama.cpp";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs =
    { nixpkgs, ... }:
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
        gbrain-embed = pkgs.callPackage ./package.nix { };
        default = gbrain-embed;
      });

      # NixOS and nix-darwin modules; the option tree is
      # `services.gbrain-embed.*` in both cases.
      nixosModules = {
        default = ./modules/nixos.nix;
        gbrain-embed = ./modules/nixos.nix;
      };

      darwinModules = {
        default = ./modules/darwin.nix;
        gbrain-embed = ./modules/darwin.nix;
      };
    };
}
