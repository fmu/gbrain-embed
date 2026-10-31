{ config, lib, pkgs, ... }:

let
  cfg = config.services.gbrain-embed;
  configPath = "/etc/gbrain-embed/config.json";
  logDir = "${config.system.primaryUserHome}/Library/Logs";
in
{
  imports = [ ./common.nix ];

  config = lib.mkIf cfg.enable {
    environment.etc."gbrain-embed/config.json".source = cfg.configFile;

    # A per-user agent: the proxy must run as the user who owns the gbrain
    # brain and the model file. nix-darwin requires `system.primaryUser` to be
    # set for these agents.
    launchd.user.agents.gbrain-embed = {
      command = "${cfg.package}/bin/gbrain-embed --config ${configPath} serve";
      environment = cfg.environment;
      serviceConfig = {
        KeepAlive = true;
        RunAtLoad = true;
        ProcessType = "Background";
        ThrottleInterval = 10;
        StandardOutPath = "${logDir}/gbrain-embed.log";
        StandardErrorPath = "${logDir}/gbrain-embed.err.log";
      };
    };
  };
}
