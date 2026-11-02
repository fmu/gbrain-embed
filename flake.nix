{
  description = "On-demand Qwen3 embeddings and reranking over an OpenAI-compatible endpoint, served by llama.cpp";

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
        qwen3-embed = pkgs.callPackage ./package.nix { };
        default = qwen3-embed;
      });

      # NixOS and nix-darwin modules; the option tree is
      # `services.qwen3-embed.*` in both cases.
      nixosModules = {
        default = ./modules/nixos.nix;
        qwen3-embed = ./modules/nixos.nix;
      };

      darwinModules = {
        default = ./modules/darwin.nix;
        qwen3-embed = ./modules/darwin.nix;
      };
    };
}
