"""Chat in the terminal through the shared MLX Chat coordinator.

Installed by install.sh as `mlx-omarchy-demo`. This is the terminal face
of MLX Chat: it attaches to the same loopback coordinator, qualified pair
selection, and pinned catalog models as the desktop web application. It
never loads weights directly and takes no Hugging Face model id — model
choice is a qualified catalog pair (`--pair everyday|quality`), downloaded
after explicit `--yes` approval on first use.

    mlx-omarchy-demo --pair everyday --yes   # scripted first turn, then interactive chat
    mlx-omarchy-demo --pair everyday --yes --once
    mlx-omarchy-demo                         # attach to the already-set-up pair
"""

import argparse
import sys

FIRST_PROMPT = "In three sentences, explain what makes Apple Silicon unusual for running language models on Linux."


def coordinator_argv(argv=None):
    """Translate demo flags onto `mlx_omarchy_assistant` terminal arguments.

    Pure argv mapping: no coordinator import, no mlx import, no download.
    The scripted first prompt keeps the demo's one-turn preview; `--once`
    stops after it. `--yes`/`--pair` cross-validation belongs to the
    shared CLI, which owns the setup flow.
    """
    parser = argparse.ArgumentParser(
        prog="mlx-omarchy-demo",
        description="Terminal chat on the Apple GPU through the shared MLX Chat coordinator.",
    )
    parser.add_argument("--pair", choices=("everyday", "quality"),
                        help="qualified model pair to set up (downloads after --yes)")
    parser.add_argument("--yes", action="store_true",
                        help="approve downloading the selected pair")
    parser.add_argument("--prompt", default=FIRST_PROMPT, help="scripted first message")
    parser.add_argument("--once", action="store_true", help="exit after the scripted first message")
    parser.add_argument("--max-tokens", type=int, help="explicit output allowance; otherwise size it for the task")
    args = parser.parse_args(argv)

    forwarded = ["--terminal", "--prompt", args.prompt]
    if args.max_tokens is not None:
        forwarded.extend(["--max-tokens", str(args.max_tokens)])
    if args.once:
        forwarded.append("--once")
    if args.pair:
        forwarded.extend(["--pair", args.pair])
    if args.yes:
        forwarded.append("--yes")
    return forwarded


def main(argv=None):
    from mlx_omarchy_assistant.__main__ import main as assistant_main

    return assistant_main(coordinator_argv(argv))


if __name__ == "__main__":
    sys.exit(main())
