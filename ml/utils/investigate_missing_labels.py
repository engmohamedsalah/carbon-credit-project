import os
import glob
import traceback
from datetime import datetime
from data_preparation import create_change_label_from_stack

s2_stack_dir = "ml/data/prepared/s2_stacks"
change_label_dir = "ml/data/prepared/change_labels"
hansen_dir = "ml/data/hansen_downloads"

# Find all *_stack.tif files
s2_stacks = [f for f in os.listdir(s2_stack_dir) if f.endswith('_stack.tif')]
# Find all change labels
change_labels = set(f for f in os.listdir(change_label_dir) if f.endswith('.tif'))

# Find required Hansen files
hansen_files = glob.glob(os.path.join(hansen_dir, '*.tif'))
hansen_treecover = next((f for f in hansen_files if 'treecover' in f), None)
hansen_lossyear = next((f for f in hansen_files if 'lossyear' in f), None)
hansen_datamask = next((f for f in hansen_files if 'datamask' in f), None)

if not (hansen_treecover and hansen_lossyear and hansen_datamask):
    print("Error: Missing Hansen files in", hansen_dir)
    exit(1)

recovered = []
failed = []

def get_label_name_from_stack(stack):
    # Extract date from stack filename
    try:
        date_str = stack.split("_")[2].split("T")[0]
        label_name = f"{date_str}_change_label.tif"
        return label_name
    except Exception:
        return None

for stack in s2_stacks:
    label_name = get_label_name_from_stack(stack)
    if not label_name:
        print(f"  Could not parse date from {stack}, skipping.")
        continue
    if label_name in change_labels:
        continue  # Already has label
    stack_path = os.path.join(s2_stack_dir, stack)
    print(f"Attempting to create label for {stack}...")
    try:
        result = create_change_label_from_stack(stack_path, hansen_treecover, hansen_lossyear, hansen_datamask, change_label_dir)
        if result and os.path.exists(os.path.join(change_label_dir, label_name)):
            print(f"  Success: Label created for {stack}")
            recovered.append(stack)
        else:
            print(f"  Failed: No label created for {stack}")
            failed.append((stack, "No label created"))
    except Exception as e:
        print(f"  Error processing {stack}: {e}")
        traceback.print_exc()
        failed.append((stack, str(e)))

print(f"\nSummary: {len(recovered)} labels recovered, {len(failed)} still missing.")
if failed:
    print("Failures:")
    for stack, reason in failed:
        print(f"  - {stack}: {reason}") 