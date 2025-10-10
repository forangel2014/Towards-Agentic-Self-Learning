# Copyright (c) 2025 RedNote Authors. All Rights Reserved.

import sys


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "rl":
        from .verl.cli import main as rl_main

        sys.argv.pop(1)
        rl_main()

    elif len(sys.argv) > 1 and sys.argv[1] in ("ray", "rlx"):
        from .verl.cli import tool as rl_tool

        sys.argv.pop(1)
        rl_tool()
    else:
        from .tuner.cli import main as tuner_main

        tuner_main()


if __name__ == "__main__":
    main()
