from pathlib import Path
import json
import csv

def jsonl_to_csv(directory_path: str) -> None:
    jsonl_path = Path(directory_path) / "checkpoints" / "results.jsonl"
    csv_path = Path(directory_path) / "summary_eval.csv"
    if not jsonl_path.exists():
        print(f"Could not find {jsonl_path}")
        return
    with jsonl_path.open('r') as jsonl_file, csv_path.open('w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        headers_written = False
        for line in jsonl_file:
            data = json.loads(line)
            if not headers_written:
                writer.writerow(data.keys())
                headers_written = True
            writer.writerow(data.values())

def apply_jsonl_to_csv_to_subdirectories(directory_path: str) -> None:
    base_path = Path(directory_path)
    for subdirectory in base_path.iterdir():
        if subdirectory.is_dir():
            jsonl_to_csv(str(subdirectory))

if __name__ == "__main__":
    for directory in ["exps/24-04-03-sidekick_runs_debug", "exps/24-04-04-cosine_warmup", "exps/24-04-04-sidekick_runs_rerun_cooldown", "exps/24-04-05-cosine_warmup_v2", "exps/24-04-05-sidekick_runs_rerun_const"]:
        apply_jsonl_to_csv_to_subdirectories(directory)
