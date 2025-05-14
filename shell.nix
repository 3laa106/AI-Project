{pkgs ? import <nixpkgs> {}}:

pkgs.mkShell  {
  packages = with pkgs; [ python3 python312Packages.numpy python312Packages.tkinter python312Packages.pillow pyright];
}
