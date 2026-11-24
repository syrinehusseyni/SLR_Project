"""Fabrique de petits PDF valides pour les tests (aucune bibliothèque nécessaire)."""


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(pages: list[str]) -> bytes:
    """Un PDF par liste de textes (une page par élément). Un texte vide donne une page sans texte (comme un scan)."""
    n = len(pages)
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(n))
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {n} >>",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for i, text in enumerate(pages):
        objects.append(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {5 + 2 * i} 0 R >>"
        )
        if text.strip():
            lines = [
                f"BT /F1 12 Tf 50 {740 - 16 * k} Td ({_escape(line)}) Tj ET"
                for k, line in enumerate(text.split("\n"))
            ]
            stream = "\n".join(lines)
        else:
            stream = "0.8 g 50 50 200 200 re f"  # un dessin, aucun texte
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("latin-1")
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode("latin-1")
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("latin-1")
    return bytes(out)


TITLE = "Automated title and abstract screening for scoping reviews using the GPT-4 Large Language Model"
BODY = (
    "Scoping reviews require intensive human effort to screen large numbers of scholarly sources. "
    "This manuscript introduces a package that uses a large language model to screen sources automatically. "
    "Validation against consensus human reviewer decisions showed good sensitivity and specificity overall."
)


def good_pdf(title: str = TITLE) -> bytes:
    """Un PDF avec du vrai texte et le titre sur la première page."""
    return make_pdf([title + "\n" + BODY, BODY, BODY])