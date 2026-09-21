from pathlib import Path
import sys
from rich import print
import simulation_code.SourcePy.record as record

def repair_and_create_archives(root_path: str = "."):
    root_dir = Path(root_path)
    
    pickle_dirs = set(p.parent for p in root_dir.rglob("*.pickle"))
    
    if not pickle_dirs:
        print("[yellow]No directories with .pickle files found.[/yellow]")
        return

    print(f"[blue]Found {len(pickle_dirs)} directories to process in '{root_dir.resolve()}'...[/blue]")

    for run_dir in pickle_dirs:
        folder_name = run_dir.name
        h5_file = run_dir / f"{folder_name}.h5"
        pickle_files = list(run_dir.glob("*.pickle"))
        
        should_archive = not h5_file.exists()
        if not should_archive and pickle_files:
            latest_pickle = max(pickle_files, key=lambda f: f.stat().st_mtime)
            if latest_pickle.stat().st_mtime > h5_file.stat().st_mtime:
                should_archive = True

        if should_archive:
            print(f"[green]Archiving/Updating:[/green] {run_dir}")
            try:
                halo_rec = record.HaloRecord(dir_data=str(run_dir))
                halo_rec.create_archive()
            except Exception as e:
                print(f"[red]Error processing {run_dir}: {e}[/red]")
        else:
            print(f"[white]Skipped (Up to date):[/white] {run_dir}")

if __name__ == "__main__":
    target_path = sys.argv[1] if len(sys.argv) > 1 else "."
    repair_and_create_archives(target_path)