import asyncio
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from datetime import datetime
import os
import sys
import traceback

asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

NOTEBOOK_ORDER = [
    "procesamiento_inicial.ipynb",
    "ner.ipynb",
    "tf-idf.ipynb",
    "topic-modeling-1.ipynb",
    "topic-modeling-2.ipynb",
    "topic-modeling-3.ipynb",
]

NOTEBOOK_DIR = ""

TIMEOUT = 900


def run_notebook(path):
    print(f"\n=== Running notebook: {path} ===")
    with open(path, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=TIMEOUT, kernel_name="python3")
    try:
        ep.preprocess(nb, {"metadata": {"path": os.path.dirname(path) or "."}})
        print(f"✔ Successfully executed: {path}")
        with open(path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        return None
    except Exception as e:
        print(f"\n❌ ERROR in {path}")
        print("-" * 80)
        traceback.print_exc()
        print("-" * 80)
        return e


def main():
    errors = []
    start_time = datetime.now()
    print(f"Started execution at {start_time:%Y-%m-%d %H:%M:%S}\n")

    for nb_name in NOTEBOOK_ORDER:
        nb_path = os.path.join(NOTEBOOK_DIR, nb_name)
        if not os.path.exists(nb_path):
            print(f"⚠ Notebook not found: {nb_path}")
            errors.append((nb_name, FileNotFoundError("Notebook not found")))
            continue
        error = run_notebook(nb_path)
        if error:
            errors.append((nb_name, error))
            print(f"❌ Stopping execution due to error in: {nb_name}")
            break

    end_time = datetime.now()
    print(f"\nFinished execution at {end_time:%Y-%m-%d %H:%M:%S}")
    print(f"Total duration: {end_time - start_time}")

    if errors:
        print("\nSummary of errors:")
        for nb_name, err in errors:
            print(f" - {nb_name}: {type(err).__name__} — {err}")
        sys.exit(1)
    else:
        print("\n✅ All notebooks executed successfully!")


if __name__ == "__main__":
    main()
