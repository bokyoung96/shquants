from __future__ import annotations

from .service import Options


class OptionsWorkflow:
    name = "options"

    def run(self, client, args) -> None:
        options = Options(client)
        if args.options_command == "raw":
            options.save_raw(date=args.date, output=args.output, limit=args.limit)
            return
        raise ValueError(f"unknown options workflow command: {args.options_command}")

