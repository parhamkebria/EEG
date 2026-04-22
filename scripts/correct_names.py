import argparse
import os
import re

PATTERN = re.compile(r"^(?P<prefix>.+)_(?P<num>\d+)\.png$", re.IGNORECASE)


def correct_names(directory):
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    grouped = {}
    for filename in files:
        match = PATTERN.match(filename)
        if not match:
            continue

        prefix = match.group("prefix")
        num = match.group("num")
        grouped.setdefault(prefix, []).append((filename, num))

    rename_pairs = []
    for prefix, entries in grouped.items():
        width = max(len(num) for _, num in entries)
        for old_name, num in entries:
            new_name = f"{prefix}_{num.zfill(width)}.png"
            if new_name != old_name:
                rename_pairs.append((old_name, new_name))

    # Two-phase rename to avoid collisions (e.g., x_1.png -> x_01.png when x_01.png exists).
    temp_pairs = []
    for i, (old_name, _) in enumerate(rename_pairs):
        old_path = os.path.join(directory, old_name)
        temp_name = f".__tmp_rename_{i}__.png"
        temp_path = os.path.join(directory, temp_name)
        os.rename(old_path, temp_path)
        temp_pairs.append((temp_name, old_name))

    for temp_name, original_old_name in temp_pairs:
        final_name = next(new for old, new in rename_pairs if old == original_old_name)
        temp_path = os.path.join(directory, temp_name)
        final_path = os.path.join(directory, final_name)
        os.rename(temp_path, final_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Zero-pad numeric suffixes in PNG filenames (e.g., absd_12.png -> absd_0012.png)."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directory containing PNG files. Defaults to current directory.",
    )
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    correct_names(args.directory)