"""将已构建的 Windows 便携目录打包，并生成 SHA-256 校验文件。"""
from hashlib import file_digest
from importlib.metadata import distribution
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile


def main():
    root = Path(__file__).resolve().parents[1]
    bundle = root / "dist" / "Mineswift"
    for required in ("Mineswift.exe", "_internal/models/belief_rl.pt", "_internal/ms/web_grid.js"):
        if not (bundle / required).is_file():
            raise SystemExit(f"缺少 {required}，请先运行 python -m PyInstaller Mineswift.spec")
    destination = root / "release"
    destination.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=destination) as temporary:
        archive = Path(temporary) / "Mineswift-Windows-x64.zip"
        with ZipFile(archive, "w", ZIP_DEFLATED, compresslevel=6) as out:
            for path in sorted(bundle.rglob("*")):
                if path.is_file():
                    out.write(path, Path("Mineswift") / path.relative_to(bundle))
            out.write(root / "LICENSE", "Mineswift/LICENSE")
            out.write(Path(sys.base_prefix) / "LICENSE.txt", "Mineswift/_internal/licences/python/LICENSE.txt")
            installer = distribution("pyinstaller")
            notice = next(f for f in installer.files if f.name == "COPYING.txt" and ".dist-info/" in str(f))
            out.write(installer.locate_file(notice), "Mineswift/_internal/licences/pyinstaller/COPYING.txt")
        with ZipFile(archive) as checked:
            damaged = checked.testzip()
            if damaged:
                raise RuntimeError(f"压缩包校验失败：{damaged}")
        with archive.open("rb") as stream:
            digest = file_digest(stream, "sha256").hexdigest()
        checksum = archive.with_suffix(".zip.sha256")
        checksum.write_text(f"{digest}  {archive.name}\n", encoding="ascii")
        archive.replace(destination / archive.name)
        checksum.replace(destination / checksum.name)
    print(f"{digest}  {destination / archive.name}")


if __name__ == "__main__":
    main()
