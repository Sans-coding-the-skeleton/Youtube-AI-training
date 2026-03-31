import os, glob

dataset_dir = "dataset"
jsons = set([os.path.basename(f).split('.')[0] for f in glob.glob(os.path.join(dataset_dir, '*.info.json'))])
imgs_webp = set([os.path.basename(f).split('.')[0] for f in glob.glob(os.path.join(dataset_dir, '*.webp'))])
imgs_jpg = set([os.path.basename(f).split('.')[0] for f in glob.glob(os.path.join(dataset_dir, '*.jpg'))])
imgs = imgs_webp.union(imgs_jpg)

valid_pairs = list(jsons.intersection(imgs))

# Retain exactly 1500 pairs
target = 1500
keep_ids = set(valid_pairs[:target])

# Remove everything else
all_files = glob.glob(os.path.join(dataset_dir, '*.*'))
removed = 0
for f in all_files:
    base = os.path.basename(f)
    file_id = base.replace('.info.json', '').replace('.webp', '').replace('.jpg', '')
    if file_id not in keep_ids:
        os.remove(f)
        removed += 1

print(f"Kept {target} pairs. Removed {removed} files.")
print("Final JSON count:", len(glob.glob(os.path.join(dataset_dir, '*.info.json'))))
print("Final IMG count:", len(glob.glob(os.path.join(dataset_dir, '*.webp'))) + len(glob.glob(os.path.join(dataset_dir, '*.jpg'))))
