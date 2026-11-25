from importlib.metadata import distribution, PackageNotFoundError
from pathlib import Path

root = Path(SPECPATH)
datas = [(str(root / "ms" / "web_grid.js"), "ms")]
for name in ("belief_rl.pt", "rl_guess.pt"):
    datas.append((str(root / "models" / name), "models"))
for name in ("numpy", "torch", "opencv-python-headless", "pillow", "comtypes", "mss",
             "playwright", "greenlet", "pyee", "sympy", "mpmath", "networkx",
             "fsspec", "jinja2", "markupsafe", "filelock", "typing_extensions"):
    try:
        dist = distribution(name)
    except PackageNotFoundError:
        continue
    for file in dist.files or ():
        if ".dist-info" in str(file) and file.name.lower().startswith(("license", "copying", "notice")):
            datas.append((str(dist.locate_file(file)), "licences/" + name))

a = Analysis(["packaging/windows_entry.py"], pathex=[str(root)], datas=datas, binaries=[],
             hiddenimports=[], excludes=["pytest", "matplotlib", "scipy", "pandas", "IPython",
                                        "torchvision", "torchaudio", "tensorboard"])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Mineswift", console=False, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name="Mineswift", upx=False)
