import json
from argparse import ArgumentParser

testing_matrix_ghact = {
    "python-version": ["3.7", "3.8", "3.9", "3.10", "3.11"],
    "os": ["windows-latest", "macos-latest", "ubuntu-latest"],
    "exclude": [{"os": "windows-latest", "python-version": "3.11"}],
}

tutorial_testing_matrix_ghact = {
    "python-version": ["3.7", "3.8", "3.9", "3.10", "3.11"],
    "os": ["macos-latest", "ubuntu-latest"],
}

deploy_matrix_ghact = {"python-version": ["3.8"], "os": ["ubuntu-latest"]}

if __name__ == "__main__":
    info = {
        "testing_matrix": testing_matrix_ghact,
        "tutorial_testing_matrix": tutorial_testing_matrix_ghact,
        "deploy_matrix": deploy_matrix_ghact,
    }
    parser = ArgumentParser(description="Export info using JSON output")
    parser.add_argument("info_type", choices=list(info.keys()))
    args = parser.parse_args()

    print(json.dumps(info[args.info_type]))
