{
  description = "Flake for hackathon";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    unstable-nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    {
      self,
      nixpkgs,
      unstable-nixpkgs,
    }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      unstable = import unstable-nixpkgs { inherit system; };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = [
          pkgs.python314
          pkgs.uv
          pkgs.just
        ];

        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
          pkgs.stdenv.cc.cc.lib
          pkgs.zlib
        ];
      };
    };
}
