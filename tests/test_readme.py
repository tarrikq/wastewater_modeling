from pathlib import Path

from wdtk import readme

ROOT = Path(__file__).resolve().parents[1]


def test_currency_escaped_math_kept() -> None:
    text = "from $639 to $924, with $h_f = f L/D$ and $10^{-9}$: done\n$$V = kCR$$"
    assert readme.escape_dollars(text) == (
        r"from \$639 to \$924, with $h_f = f L/D$ and $10^{-9}$: done" + "\n$$V = kCR$$"
    )
    assert readme.escape_dollars(r"already \$5") == r"already \$5"


def test_relative_links_point_at_github() -> None:
    text = "[a](data/x.yaml) [b](src/wdtk/) [c](https://example.com) [d](#top)"
    assert readme.absolute_links(text) == (
        f"[a]({readme.REPO_URL}/blob/master/data/x.yaml) "
        f"[b]({readme.REPO_URL}/tree/master/src/wdtk/) [c](https://example.com) [d](#top)"
    )


def test_blocks_split_images_and_drop_banner() -> None:
    text = f"# T\n\n{readme.APP_BANNER}(x)**\n\nIntro\n![Fig](a.png)\n![Fig 2](b.png)\nEnd"
    assert readme.blocks(text) == [
        readme.Block("markdown", "# T\n\n\nIntro"),
        readme.Block("image", "a.png", "Fig"),
        readme.Block("image", "b.png", "Fig 2"),
        readme.Block("markdown", "End"),
    ]


def test_repository_readme_renders() -> None:
    blocks = readme.blocks((ROOT / "README.md").read_text(encoding="utf-8"))
    images = [b for b in blocks if b.kind == "image"]
    assert images
    assert all((ROOT / b.body).is_file() for b in images)
    markdown = "\n".join(b.body for b in blocks if b.kind == "markdown")
    assert readme.APP_BANNER not in markdown
    assert "](data/" not in markdown and "](reports/" not in markdown
