import argparse
import os
import re

module = "pydmd"
meta_file = os.path.join(module, "meta.py")
version_line = r'__version__.*=.*"(.+?)"'


class Version:
    def __init__(self, major, minor, patch, date_patch=None):
        self.major = major
        self.minor = minor
        self.patch = patch
        self.date_patch = date_patch

    def __str__(self):
        version_string = f"{self.major}.{self.minor}.{self.patch}"
        if self.date_patch:
            version_string = f"{version_string}.{self.date_patch}"
        return version_string


def get_version():
    with open(meta_file, "r", encoding="UTF-8") as fp:
        content = fp.read()

    try:
        found = re.search(r'__version__.*=.*"(.+?)"', content).group(1)
    except AttributeError:
        pass

    version = re.split(r"[-\.]", found)
    v = Version(*version)
    return v


def set_version(version):
    with open(meta_file, "r", encoding="UTF-8") as fp:
        content = fp.read()

    line_string = '__version__ = "{}"'.format(version)
    text_after = re.sub('__version__.*=.*"(.+?)"', line_string, content)

    with open(meta_file, "w", encoding="UTF-8") as fp:
        fp.write(text_after)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manipulate Version")

    subparsers = parser.add_subparsers(dest="command")

    get_ = subparsers.add_parser(
        "get", help="Get information about current version"
    )
    set_ = subparsers.add_parser("set", help="Set version")
    flags = set_.add_mutually_exclusive_group(required=False)
    flags.add_argument("--only-major", action="store_true")
    flags.add_argument("--only-minor", action="store_true")
    flags.add_argument("--only-patch", action="store_true")
    flags.add_argument("--only-date", action="store_true")
    set_.add_argument("version", nargs="+", action="store")

    args = parser.parse_args()

    if args.command == "get":
        print(get_version())
    elif args.command == "set":
        if args.only_major:
            current_version = get_version()
            current_version.major = args.version[0]
            set_version(current_version)
        elif args.only_minor:
            current_version = get_version()
            current_version.minor = args.version[0]
            set_version(current_version)
        elif args.only_patch:
            current_version = get_version()
            current_version.patch = args.version[0]
            set_version(current_version)
        elif args.only_date:
            current_version = get_version()
            current_version.date_patch = args.version[0]
            set_version(current_version)
        elif len(args.version) in [3, 4]:
            set_version(Version(*args.version))
        else:
            raise RuntimeError
