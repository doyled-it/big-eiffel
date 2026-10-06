"""Build the deployable static site for Cloudflare Pages (or any static host).

Injects the chart data into the page template and wraps the fragment in a full
HTML document, writing ``site/index.html``. The template (``web/index.template.html``)
carries a ``__DATA__`` placeholder that is replaced with ``outputs/chartdata.json``.

Usage:
    uv run python scripts/build_site.py      # or: make site
"""

from __future__ import annotations

import json
from pathlib import Path

from frb import config as C

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "web" / "index.template.html"
SITE = ROOT / "site"


def main() -> None:
    data = json.dumps(json.load(open(C.OUTPUTS / "chartdata.json")), separators=(",", ":"))
    fragment = TEMPLATE.read_text().replace("__DATA__", data)
    if fragment.count("__DATA__"):
        raise SystemExit("template still has an unfilled __DATA__ placeholder")

    # The template is a fragment: head elements (title, meta, links, style) up to
    # the first </style>, then the body (controls, content, scripts). Wrap it in a
    # proper document so it stands alone outside the artifact host.
    head, body = fragment.split("</style>", 1)
    html = (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">\n'
        f"{head}</style>\n"
        "</head>\n"
        f"<body>\n{body.lstrip()}\n</body>\n"
        "</html>\n"
    )
    SITE.mkdir(exist_ok=True)
    out = SITE / "index.html"
    out.write_text(html)
    print(f"wrote {out} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
