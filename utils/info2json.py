import json

config = {
    'os': [
        "windows-latest",
        "macos-latest",
        "ubuntu-latest"
    ]
}
print(json.dumps(config))
