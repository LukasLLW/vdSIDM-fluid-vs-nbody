import sys
import shutil
from pathlib import Path
from reader_code.gravothermal_reader_code.run_archives import repair_and_create_archives

if __name__ == "__main__":
    target_path = sys.argv[1] if len(sys.argv) > 1 else "."
    
    step = None
    if len(sys.argv) > 2:
        try:
            step = int(sys.argv[2])
        except ValueError:
            step = None

    out_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else None

    if step is not None and step > 0:
        target_dir = Path(target_path).resolve()
        
        pickle_dirs = set(p.parent for p in target_dir.rglob("*.pickle"))

        if not pickle_dirs:
            print("[yellow]No directories with .pickle files found.[/yellow]")
        else:
            for p_dir in pickle_dirs:
                if out_dir:
                    rel_path = p_dir.relative_to(target_dir)
                    dest_dir = out_dir.resolve() / rel_path
                    dest_dir.mkdir(parents=True, exist_ok=True)

                    copied_pickles = []
                    pickle_files = sorted(list(p_dir.glob("*.pickle")))
                    for pickle_file in pickle_files[::step]:
                        target_file = dest_dir / pickle_file.name
                        shutil.copy2(pickle_file, target_file)
                        copied_pickles.append(target_file)

                    for h5_file in p_dir.glob("*.h5"):
                        shutil.copy2(h5_file, dest_dir / h5_file.name)

                    repair_and_create_archives(str(dest_dir))

                    for copied_file in copied_pickles:
                        if copied_file.exists():
                            copied_file.unlink()

                else:
                    dest_dir = p_dir / "temp"
                    dest_dir.mkdir(parents=True, exist_ok=True)

                    pickle_files = sorted(list(p_dir.glob("*.pickle")))
                    for pickle_file in pickle_files[::step]:
                        shutil.copy2(pickle_file, dest_dir / pickle_file.name)

                    for h5_file in p_dir.glob("*.h5"):
                        shutil.copy2(h5_file, dest_dir / h5_file.name)

                    repair_and_create_archives(str(dest_dir))
    else:
        repair_and_create_archives(target_path)
