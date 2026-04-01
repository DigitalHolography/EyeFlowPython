"""Allow `python -m eyeflow` to behave like the CLI entrypoint."""

from eyeflow.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
