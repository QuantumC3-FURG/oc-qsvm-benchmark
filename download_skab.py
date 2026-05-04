"""
download_skab.py — Download SKAB others/5.csv through 14.csv
=============================================================
Usage:
    python download_skab.py

Creates data/skab/others/ and saves the CSV files there.
"""
import os
import time
import urllib.request

# SKAB repository raw-content URLs (GitHub).
# Dataset source: https://github.com/waico/SKAB/tree/master/data/other/
BASE_URL = "https://raw.githubusercontent.com/waico/SKAB/master/data/other"
FILES    = list(range(5, 15))   # files 5.csv … 14.csv (the "others" subset)

# Local destination
OUT_DIR  = os.path.join("data", "skab", "others")

def download_file(url: str, dest: str) -> bool:
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Destination: {os.path.abspath(OUT_DIR)}\n")
    ok, fail = [], []

    for n in FILES:
        fname = f"{n}.csv"
        dest  = os.path.join(OUT_DIR, fname)

        if os.path.exists(dest):
            print(f"  [OK – cached] {fname}")
            ok.append(n)
            continue

        url = f"{BASE_URL}/{fname}"
        print(f"  Downloading {url} …", end=" ", flush=True)

        if download_file(url, dest):
            size = os.path.getsize(dest)
            print(f"OK ({size:,} bytes)")
            ok.append(n)
        else:
            fail.append(n)
        time.sleep(0.3)

    print(f"\nCompleted: {len(ok)} files OK | {len(fail)} failures")
    if fail:
        print(f"  Failed: {fail}")
        print("  Manual download URLs:")
        for n in fail:
            print(f"    {BASE_URL}/{n}.csv  →  {OUT_DIR}/{n}.csv")


if __name__ == "__main__":
    main()