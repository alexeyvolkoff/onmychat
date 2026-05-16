import requests
import sys

def check_xml(url, token):
    headers = {"Authorization": f"token:{token}"}
    r = requests.get(f"{url}?list", headers=headers)
    if r.status_code == 200:
        print(r.text)
    else:
        print(f"Error: {r.status_code}")

if __name__ == "__main__":
    url = sys.argv[1]
    token = "bdeeccd12de74106cbcf5d4d035ab1ef" # OMD_TOKEN from config.ini
    check_xml(url, token)
