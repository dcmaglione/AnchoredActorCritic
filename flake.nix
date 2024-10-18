{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nixgl.url = "github:guibou/nixGL";
    nixgl.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = {self, nixpkgs, nixgl, ... }@inp:
    let
      l = nixpkgs.lib // builtins;
      supportedSystems = [ "x86_64-linux" "aarch64-darwin" ];
      forAllSystems = f: l.genAttrs supportedSystems
        (system: f system (import nixpkgs {inherit system;
          config.permittedInsecurePackages = [
              "python3.11-youtube-dl-2021.12.17" # moviepy uses this and gymnasium uses moviepy
            ];
        # overlays=[nixgl.overlay]; config.allowUnfree=true;
          # config.cudaSupport = true; config.cudaCapabilities = [ "8.6" ];
          }));
      
    in
    {
      # enter this python environment by executing `nix shell .`
      devShell = forAllSystems (system: pkgs:
        let
            python = pkgs.python311.withPackages (p: with p;[
              numpy pygame pybullet matplotlib gymnasium tensorflow tqdm keras pybox2d seaborn scipy
              (callPackage ./mujoco-py.nix {})
            ]);
        in pkgs.mkShell {
            buildInputs = [
                # pkgs.nixgl.auto.nixGLDefault
                python
            ];
            shellHook = ''
              export PYTHONPATH=$PYTHONPATH:$(pwd)
              # export LD_LIBRARY_PATH=${pkgs.wayland}/lib:$LD_LIBRARY_PATH:/run/opengl-driver/lib
            '';
          }
        );
    };
}
