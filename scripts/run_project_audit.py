"""Print the project audit location and verify it exists."""

from pathlib import Path


def main() -> None:
    path = Path("reports/project_audit.md")
    if not path.exists():
        raise SystemExit("reports/project_audit.md is missing")
    print(path.resolve())


if __name__ == "__main__":
    main()
