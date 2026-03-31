import os
import json
import csv

dataset_dir = "dataset"
output_file = "dataset.csv"

fields = [
    "id", "title", "description", "duration", "view_count", 
    "like_count", "comment_count", "upload_date", "channel", 
    "channel_id", "channel_follower_count", "categories", "tags", "chapters", "thumbnail", "webpage_url"
]

def extract_metadata(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    row = {}
    for field in fields:
        val = data.get(field, "")
        
        # Handle lists like categories and tags
        if isinstance(val, list):
            # If it's chapters, keep as JSON string
            if field == "chapters":
                val = json.dumps(val)
            else:
                val = ", ".join(val)
        
        row[field] = val
    return row

def main():
    json_files = [f for f in os.listdir(dataset_dir) if f.endswith(".info.json")]
    print(f"Found {len(json_files)} JSON files.")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        
        for i, filename in enumerate(json_files):
            json_path = os.path.join(dataset_dir, filename)
            try:
                metadata = extract_metadata(json_path)
                writer.writerow(metadata)
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1} files...")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print(f"Sucessfully created {output_file}")

if __name__ == "__main__":
    main()
