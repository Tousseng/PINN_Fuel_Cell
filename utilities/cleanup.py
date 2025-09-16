import os
import tempfile

def clean_autograph_cache(verbose: bool = True):
    """
    Automatically deletes all generated AutoGraph-files by TensorFlow
    from the temporary directory TEMP (z.B. __autograph_generated_file*.py).

    Parameters:
        verbose: Toggle if the name of deleted files should be shown.
    """
    tmp_dir: str = tempfile.gettempdir()
    pycache_dir: str = os.path.join(tmp_dir, "__pycache__")
    removed: int = 0

    removed += remove_from_dir(
        directory=tmp_dir, ending=".py", verbose=verbose
    )
    removed += remove_from_dir(
        directory=pycache_dir, ending=".pyc", verbose=verbose
    )

    if verbose:
        print(f"Total deletions: {removed} AutoGraph-files.")

def remove_from_dir(directory: str, ending: str, verbose: bool) -> int:
    removed: int = 0
    for fname in os.listdir(directory):
        if fname.startswith("__autograph_generated_file") and fname.endswith(ending):
            fpath = os.path.join(directory, fname)
            try:
                os.remove(fpath)
                removed += 1
                if verbose:
                    print(f"Deleted: {fname}")
            except FileNotFoundError as e:
                print(f"Couldn't delete {fname}: {e}")
    return removed

if __name__ == "__main__":
    clean_autograph_cache()