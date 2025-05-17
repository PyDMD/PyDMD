import json
from argparse import ArgumentParser

testing_matrix_ghact = {
    "python-version": ["3.9", "3.10", "3.11", "3.12"],
    "os": ["windows-latest", "macos-15", "ubuntu-latest"],
    "numpy-version": ["1.26.4", "2.0.2"],
    "exclude": [
        {"os": "windows-latest", "python-version": "3.11"},
    ],
}

deploy_matrix_ghact = {"python-version": ["3.8"], "os": ["ubuntu-latest"]}

if __name__ == "__main__":
    info = {
        "testing_matrix": testing_matrix_ghact,
        "deploy_matrix": deploy_matrix_ghact,
    }
    parser = ArgumentParser(description="Export info using JSON output")
    parser.add_argument("info_type", choices=list(info.keys()))
    args = parser.parse_args()

    print(json.dumps(info[args.info_type]))
