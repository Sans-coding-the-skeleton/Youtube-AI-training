import os
import glob
import yt_dlp

dataset_dir = "dataset"
os.makedirs(dataset_dir, exist_ok=True)

# Broad collection of specific queries to get a diverse dataset
queries = [
    "ytsearch50:minecraft gameplay",
    "ytsearch50:python tutorial for beginners",
    "ytsearch50:funny cat compilation",
    "ytsearch50:latest smartphone review",
    "ytsearch50:travel vlog japan",
    "ytsearch50:easy healthy recipes",
    "ytsearch50:makeup tutorial everyday",
    "ytsearch50:car review 2024",
    "ytsearch50:home workout no equipment",
    "ytsearch50:street food india",
    "ytsearch50:how to build a pc",
    "ytsearch50:asmr sleep",
    "ytsearch50:music video pop 2024",
    "ytsearch50:stand up comedy special",
    "ytsearch50:history documentary brief",
    "ytsearch50:woodworking projects",
    "ytsearch50:learning javascript",
    "ytsearch50:financial freedom tips",
    "ytsearch50:drawing tutorial real time",
    "ytsearch50:movie trailer 2024",
    "ytsearch50:top 10 scary stories",
    "ytsearch50:unboxing random stuff",
    "ytsearch50:day in the life software engineer",
    "ytsearch50:drone footage 4k",
    "ytsearch50:cleaning motivation",
    "ytsearch50:abandoned places exploration",
    "ytsearch50:guitar cover acoustic",
    "ytsearch50:baking sourdough bread",
    "ytsearch50:magic tricks revealed",
    "ytsearch50:science experiments for kids",
    "ytsearch50:astrophotography tutorial",
    "ytsearch50:camping in the rain"
]

ydl_opts = {
    'skip_download': True,
    'writeinfojson': True,
    'writethumbnail': True,
    'outtmpl': f'{dataset_dir}/%(id)s.%(ext)s',
    'ignoreerrors': True,
    'quiet': True,
    'no_warnings': True,
    'extract_flat': False
}

def count_valid_pairs():
    jsons = set([f.split('.')[0] for f in os.listdir(dataset_dir) if f.endswith('.info.json')])
    imgs = set([f.split('.')[0] for f in os.listdir(dataset_dir) if f.endswith('.jpg') or f.endswith('.webp')])
    return len(jsons.intersection(imgs))

def cleanup_orphans():
    jsons = set([f.split('.')[0] for f in os.listdir(dataset_dir) if f.endswith('.info.json')])
    imgs = set([f.split('.')[0] for f in os.listdir(dataset_dir) if f.endswith('.jpg') or f.endswith('.webp')])
    
    valid_ids = jsons.intersection(imgs)
    deleted = 0
    for f in os.listdir(dataset_dir):
        file_id = f.split('.')[0]
        if file_id not in valid_ids:
            os.remove(os.path.join(dataset_dir, f))
            deleted += 1
    if deleted > 0:
        print(f"Cleaned up {deleted} orphaned files (missing json or missing image).")

print("Starting deep extraction to reach exactly 1000 pairs...")
target = 1000

# Cleanup initially because previously we had 485 jsons and 480 webps
cleanup_orphans()
current_count = count_valid_pairs()
print(f"Initial valid pairs: {current_count}")

try:
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for query in queries:
            if current_count >= target:
                break
            
            print(f"\\n--- Fetching: {query} (Current count: {current_count}/{target}) ---")
            ydl.download([query])
            
            cleanup_orphans()
            current_count = count_valid_pairs()

except KeyboardInterrupt:
    print("Download interrupted by user.")

# Final cleanup
cleanup_orphans()
final_count = count_valid_pairs()

print("\\n=======================================================")
if final_count >= target:
    print(f"SUCCESS! We reached the target. Total valid pairs: {final_count}")
    
    # If we overshoot, let's delete random extras to be exactly 1000
    if final_count > target:
        import random
        jsons = [f for f in os.listdir(dataset_dir) if f.endswith('.info.json')]
        extras = random.sample(jsons, final_count - target)
        for extra in extras:
            file_id = extra.split('.')[0]
            # Delete json and any matching images
            for f in os.listdir(dataset_dir):
                if f.startswith(file_id + '.'):
                    os.remove(os.path.join(dataset_dir, f))
        print(f"Trimmed excess to exactly 1000 pairs.")
        final_count = count_valid_pairs()
else:
    print(f"Finished queries but fell short. Total valid pairs: {final_count}")

print(f"Final valid pair count: {final_count} jsons and {final_count} thumbnails.")
print("=======================================================\\n")
