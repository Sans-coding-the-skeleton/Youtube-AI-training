import os
import yt_dlp
import time

dataset_dir = "dataset"
os.makedirs(dataset_dir, exist_ok=True)

# Very niche queries to ensure fresh, unbanned results that we haven't hit yet
# Expanded queries for reaching 5000 records
queries = [
    "ytsearch150:news breaking headlines world",
    "ytsearch150:political debate analysis",
    "ytsearch150:economic crisis inflation explained",
    "ytsearch150:stock market trend analysis",
    "ytsearch150:fitness challenge routine",
    "ytsearch150:cooking show recipes easy",
    "ytsearch150:science documentary physics",
    "ytsearch150:history of ancient civilizations",
    "ytsearch150:philosophy big questions",
    "ytsearch150:movie review blockbusters",
    "ytsearch150:gaming walkthrough speedrun",
    "ytsearch150:indie game development devlog",
    "ytsearch150:music synthesis modular synth",
    "ytsearch150:piano cover classical",
    "ytsearch150:drone travel cinematic footage",
    "ytsearch150:woodworking diy furniture",
    "ytsearch150:electronics repair restoration",
    "ytsearch150:origami techniques professional",
    "ytsearch150:yoga for flexibility",
    "ytsearch150:space exploration nasa live",
    "ytsearch150:math tutorials university",
    "ytsearch150:learn coding python projects",
    "ytsearch150:travel vlog iceland italy",
    "ytsearch150:unboxing latest gadgets tech",
    "ytsearch150:smart home automation tour",
    "ytsearch150:mechanical keyboard building",
    "ytsearch150:3d printing projects timelapse",
    "ytsearch150:hydroponics indoor gardening",
    "ytsearch150:bread making sourdough art",
    "ytsearch150:aquarium scape setup",
    "ytsearch150:car fixing engine repair",
    "ytsearch150:shoe restoration cobbler",
    "ytsearch150:tiny house off grid tour",
    "ytsearch150:urban exploring abandoned",
    "ytsearch150:mountaineering climbing documentary",
    "ytsearch150:meditation mindfulness relaxation",
    "ytsearch150:nature sounds rain forest",
    "ytsearch150:asmr relaxation whispering",
    "ytsearch150:street food tour asia",
    "ytsearch150:top 10 scary facts",
    "ytsearch150:productivity hacks students",
    "ytsearch150:interior design home makeover",
    "ytsearch150:diy crafts for kids",
    "ytsearch150:fashion trends 2024",
    "ytsearch150:makeup tutorial everyday look",
    "ytsearch150:physics experiments university",
    "ytsearch150:biology cells under microscope",
    "ytsearch150:chemistry reactions satisfy",
    "ytsearch150:math riddles brain teasers",
    "ytsearch150:history of space flight",
    "ytsearch150:aviation documentaries planes",
    "ytsearch150:shipwreck exploration ocean",
    "ytsearch150:train travel scenic routes",
    "ytsearch150:car restoration vintage",
    "ytsearch150:motorcycle engine rebuild",
    "ytsearch150:garden design landscaping",
    "ytsearch150:beekeeping for beginners",
    "ytsearch150:wood carving techniques",
    "ytsearch150:knitting crochet patterns",
    "ytsearch150:baking cakes professional",
    "ytsearch150:chef recipes masterclass",
    "ytsearch150:street photography tips",
    "ytsearch150:portrait drawing tutorial",
    "ytsearch150:3d animation blender",
    "ytsearch150:unreal engine 5 game dev",
    "ytsearch150:linux desktop environment ricing",
    "ytsearch150:arch linux install guide",
    "ytsearch150:linux server tutorial hosting",
    "ytsearch150:ubuntu vs debian comparison",
    "ytsearch150:video game skit funny",
    "ytsearch150:fps game in real life skit",
    "ytsearch150:when you play too much rpg games skit",
    "ytsearch150:gamers in a nutshell comedy skit",
    "ytsearch150:pc master race vs console skit"
]

ydl_opts = {
    'skip_download': True,
    'writeinfojson': True,
    'writethumbnail': True,
    'outtmpl': f'{dataset_dir}/%(id)s.%(ext)s',
    'ignoreerrors': True,
    'quiet': False,
    'no_warnings': True,
    'extract_flat': False,
    'no_overwrites': True,
    'sleep_interval': 5,          # Crucial: By adding a 5-second delay between requests, YouTube drops the temp rate-limit ban
    'max_sleep_interval': 10
}

def count_valid_pairs():
    jsons = set([f.split('.')[0] for f in os.listdir(dataset_dir) if f.endswith('.info.json')])
    imgs = set([f.split('.')[0] for f in os.listdir(dataset_dir) if f.endswith('.jpg') or f.endswith('.webp')])
    return len(jsons.intersection(imgs))

current_count = count_valid_pairs()
target = 5000

print(f"Current valid YouTube pairs: {current_count}")
print("Starting extraction with human-like delays to bypass the rate limit...")

import random
new_qs = [q for q in queries if "linux" in q or "skit" in q]
old_qs = [q for q in queries if "linux" not in q and "skit" not in q]
random.shuffle(old_qs)
queries = new_qs + old_qs

try:
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for query in queries:
            if current_count >= target:
                break
                
            print(f"\\n--- Fetching: {query} (Targeting: {target}) ---")
            ydl.download([query])
            current_count = count_valid_pairs()
            
except KeyboardInterrupt:
    print("Download interrupted.")

final_count = count_valid_pairs()
print(f"\\nFinished. Final valid pair count: {final_count} pairs.")
