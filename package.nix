{
  lib,
  writeShellApplication,
  bun,
  curl,
  coreutils,
}:

writeShellApplication {
  name = "qwen3-embed";
  runtimeInputs = [
    bun
    curl
    coreutils
  ];
  text = ''
    exec ${bun}/bin/bun ${./src}/cli.ts "$@"
  '';
  meta = {
    description = "On-demand local Qwen3 embeddings and reranking, served by llama.cpp";
    homepage = "https://github.com/fmu/qwen3-embed";
    mainProgram = "qwen3-embed";
    platforms = lib.platforms.unix;
  };
}
