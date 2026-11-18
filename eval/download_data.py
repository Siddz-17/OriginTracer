"""Fetch the AVeriTeC dev split (500 labelled real-world claims, CC BY-NC 4.0) into eval/data/.
The file is not committed to this repository because of its licence."""
import os
import urllib.request

URL = "https://raw.githubusercontent.com/MichSchli/AVeriTeC/main/data/dev.json"
dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "averitec_dev.json")
os.makedirs(os.path.dirname(dest), exist_ok=True)
urllib.request.urlretrieve(URL, dest)
print(f"saved {os.path.getsize(dest):,} bytes to {dest}")
