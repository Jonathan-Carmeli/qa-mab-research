"""Convert two markdown files to PDFs via weasyprint."""
import os
import markdown
from weasyprint import HTML, CSS

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(HERE, "..")
OUT_DIR = "/tmp"

CSS_STYLE = """
@page {
  size: A4;
  margin: 1.8cm;
  @bottom-center { content: counter(page) " / " counter(pages); font-size: 9pt; color: #888; }
}
body {
  font-family: 'Helvetica', 'Arial', sans-serif;
  font-size: 10.5pt;
  line-height: 1.45;
  color: #222;
}
h1 { font-size: 22pt; color: #1a1a1a; border-bottom: 2px solid #444; padding-bottom: 6px; margin-top: 0; }
h2 { font-size: 15pt; color: #2a2a2a; border-bottom: 1px solid #ccc; padding-bottom: 3px; margin-top: 22px; }
h3 { font-size: 12pt; color: #333; margin-top: 16px; }
h4 { font-size: 11pt; color: #444; }
p, li { text-align: justify; }
code { font-family: 'Menlo', 'Consolas', 'Courier New', monospace; font-size: 9.5pt; background: #f5f5f5; padding: 1px 4px; border-radius: 3px; }
pre { background: #f5f5f5; border: 1px solid #ddd; padding: 10px; border-radius: 4px; overflow-x: auto; font-size: 9pt; line-height: 1.35; page-break-inside: avoid; }
pre code { background: transparent; padding: 0; font-size: inherit; }
table { border-collapse: collapse; margin: 10px 0; font-size: 9.5pt; page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 4px 8px; text-align: left; }
th { background: #eee; font-weight: 600; }
blockquote { border-left: 4px solid #888; margin: 10px 0; padding: 4px 12px; color: #444; background: #f8f8f8; }
hr { border: none; border-top: 1px solid #bbb; margin: 18px 0; }
a { color: #1565c0; text-decoration: none; }
"""

def convert(src_md, dst_pdf, title):
    with open(src_md) as f:
        md = f.read()
    html_body = markdown.markdown(
        md,
        extensions=["tables", "fenced_code", "toc", "sane_lists"],
        output_format="html5",
    )
    full_html = f"""<!doctype html><html dir="auto"><head><meta charset="utf-8"><title>{title}</title></head><body>{html_body}</body></html>"""
    HTML(string=full_html).write_pdf(dst_pdf, stylesheets=[CSS(string=CSS_STYLE)])
    print(f"Wrote {dst_pdf}")


def main():
    convert(os.path.join(SRC_DIR, "README.md"),
            os.path.join(OUT_DIR, "qubo_oneshot_README.pdf"),
            "DIAMOND-QUBO vs NB3R — Report")
    convert(os.path.join(SRC_DIR, "ALGORITHM.md"),
            os.path.join(OUT_DIR, "qubo_oneshot_ALGORITHM.pdf"),
            "DIAMOND-QUBO — Algorithm Reference")


if __name__ == "__main__":
    main()
