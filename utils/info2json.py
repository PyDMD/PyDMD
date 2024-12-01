import json
from argparse import ArgumentParser

# TODO: Monitor https://github.com/federicocarboni/setup-ffmpeg/issues/21
macos_version = "macos-13"

testing_matrix_ghact = {
    "python-version": ["3.8", "3.9", "3.10", "3.11", "3.12"],
    "os": ["windows-latest", macos_version, "ubuntu-latest"],
    "exclude": [
        {"os": "windows-latest", "python-version": "3.11"},
        {"python-version": "3.8", "numpy-version": "2.0.2"},
    ],
    "numpy-version": ["1.24.4", "2.0.2"],
}

tutorial_testing_matrix_ghact = {
    "python-version": testing_matrix_ghact["python-version"],
    "os": [macos_version, "ubuntu-latest"],
    "numpy-version": testing_matrix_ghact["numpy-version"],
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
