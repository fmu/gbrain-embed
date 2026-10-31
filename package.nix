{
  lib,
  writeShellApplication,
  bun,
  curl,
  coreutils,
}:

writeShellApplication {
  name = "gbrain-embed";
  runtimeInputs = [
    bun
    curl
    coreutils
  ];
  text = ''
    exec ${bun}/bin/bun ${./src}/cli.ts "$@"
  '';
  meta = {
    description = "On-demand local embeddings for gbrain, served by llama.cpp";
    homepage = "https://github.com/fmu/gbrain-embed";
    mainProgram = "gbrain-embed";
    platforms = lib.platforms.unix;
  };
}
