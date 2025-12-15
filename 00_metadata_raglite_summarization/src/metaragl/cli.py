import argparse
import json
from .pipeline import run

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    runp = sub.add_parser("run", help="Run metadata summarization on a local doc")
    runp.add_argument("path", help="Path to a text file (demo)")

    args = ap.parse_args()

    if args.cmd == "run":
        out = run(args.path)
        print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()