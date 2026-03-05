#!/usr/bin/env python3

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape
from reportlab.lib.units import inch
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import Table, TableStyle
from reportlab.pdfgen import canvas


PAGE_WIDTH, PAGE_HEIGHT = landscape((13.333 * inch, 7.5 * inch))
MARGIN_X = 0.6 * inch
MARGIN_TOP = 0.55 * inch
TITLE_Y = PAGE_HEIGHT - MARGIN_TOP
CONTENT_TOP = PAGE_HEIGHT - 1.25 * inch
CONTENT_BOTTOM = 0.55 * inch
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN_X

BG = colors.HexColor("#F7F3EA")
TITLE = colors.HexColor("#182430")
BODY = colors.HexColor("#27313A")
ACCENT = colors.HexColor("#B44D2A")
ACCENT_2 = colors.HexColor("#366A8A")
ACCENT_3 = colors.HexColor("#709255")
BOX_FILL = colors.HexColor("#FFFDF8")
BOX_STROKE = colors.HexColor("#7A6F61")
SUBTLE = colors.HexColor("#8E867C")


@dataclass
class Slide:
    title: str
    lines: List[str]


def wrap_text(text: str, font_name: str, font_size: int, max_width: float) -> List[str]:
    words = text.split()
    if not words:
        return [""]
    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if stringWidth(candidate, font_name, font_size) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def parse_slides(md_path: Path) -> List[Slide]:
    text = md_path.read_text()
    sections = re.split(r"(?m)^---\s*$", text)
    slides = []
    for section in sections:
        lines = [line.rstrip() for line in section.strip().splitlines()]
        if not lines:
            continue
        title = ""
        body = []
        skipping_notes = False
        in_code = False
        code_lang = ""
        code_lines: List[str] = []
        for line in lines:
            if line.startswith("#"):
                if not title:
                    title = re.sub(r"^#+\s*", "", line).strip()
                else:
                    body.append(line)
                continue
            if line.strip() == "Speaker note:":
                skipping_notes = True
                continue
            if skipping_notes:
                if not line.strip():
                    skipping_notes = False
                continue
            if line.startswith("```"):
                if in_code:
                    body.append(f"```{code_lang}")
                    body.extend(code_lines)
                    body.append("```")
                    in_code = False
                    code_lang = ""
                    code_lines = []
                else:
                    in_code = True
                    code_lang = line.strip("`").strip()
                continue
            if in_code:
                code_lines.append(line)
                continue
            body.append(line)
        slides.append(Slide(title=title or "Slide", lines=body))
    return slides


def draw_background(c: canvas.Canvas, slide_num: int):
    c.setFillColor(BG)
    c.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, stroke=0, fill=1)
    c.setFillColor(colors.white)
    c.setStrokeColor(colors.white)
    c.roundRect(0.22 * inch, 0.22 * inch, PAGE_WIDTH - 0.44 * inch, PAGE_HEIGHT - 0.44 * inch, 18, stroke=0, fill=1)
    c.setStrokeColor(colors.HexColor("#E7DDD0"))
    c.setLineWidth(1)
    c.line(MARGIN_X, PAGE_HEIGHT - 0.9 * inch, PAGE_WIDTH - MARGIN_X, PAGE_HEIGHT - 0.9 * inch)
    c.setFillColor(SUBTLE)
    c.setFont("Helvetica", 10)
    c.drawRightString(PAGE_WIDTH - MARGIN_X, 0.32 * inch, f"{slide_num}")


def draw_title(c: canvas.Canvas, title: str):
    c.setFillColor(TITLE)
    c.setFont("Helvetica-Bold", 28)
    c.drawString(MARGIN_X, TITLE_Y, title)


def draw_bullets(c: canvas.Canvas, lines: List[str], y_start: float = CONTENT_TOP, font_size: int = 18):
    y = y_start
    indent = 0.28 * inch
    bullet_gap = 0.16 * inch
    for raw in lines:
        line = raw.strip()
        if not line:
            y -= 0.14 * inch
            continue
        prefix = ""
        content = line
        if line.startswith("- "):
            prefix = "•"
            content = line[2:].strip()
        elif re.match(r"^\d+\.\s+", line):
            prefix = line.split(".", 1)[0] + "."
            content = re.sub(r"^\d+\.\s+", "", line)
        else:
            prefix = ""
        wrapped = wrap_text(content, "Helvetica", font_size, CONTENT_WIDTH - indent - 0.2 * inch)
        c.setFillColor(BODY)
        if prefix:
            c.setFont("Helvetica-Bold", font_size)
            c.drawString(MARGIN_X, y, prefix)
            x_text = MARGIN_X + indent
        else:
            x_text = MARGIN_X
        c.setFont("Helvetica", font_size)
        for idx, part in enumerate(wrapped):
            c.drawString(x_text, y, part)
            y -= font_size * 1.35
        y -= bullet_gap


def draw_table_slide(c: canvas.Canvas, slide: Slide):
    draw_title(c, slide.title)
    rows = [
        ["Concern", "Sites", "Humans (legacy)", "Humans (demo)"],
        ["Authentication", "mTLS certs", "mTLS certs", "OIDC browser login"],
        ["Identity source", "cert CN + org", "cert CN + org", "IdP claims"],
        ["Lifecycle", "provisioned, long-lived", "provisioned, long-lived", "IdP-managed, dynamic"],
        ["Startup kit", "yes", "yes", "generated console profile only"],
    ]
    table = Table(rows, colWidths=[1.7 * inch, 1.7 * inch, 2.2 * inch, 2.7 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), ACCENT_2),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 14),
                ("LEADING", (0, 0), (-1, -1), 17),
                ("BACKGROUND", (0, 1), (-1, -1), BOX_FILL),
                ("GRID", (0, 0), (-1, -1), 1, BOX_STROKE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F6F9FB")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    width, height = table.wrap(0, 0)
    table.drawOn(c, MARGIN_X, CONTENT_TOP - height)
    draw_bullets(
        c,
        [
            "",
            "The important split is infrastructure identity vs human identity.",
            "This demo changes only the human plane.",
        ],
        y_start=CONTENT_TOP - height - 0.35 * inch,
        font_size=16,
    )


def draw_box(c: canvas.Canvas, x: float, y: float, w: float, h: float, label: str, fill_color=BOX_FILL, text_color=TITLE):
    c.setFillColor(fill_color)
    c.setStrokeColor(BOX_STROKE)
    c.setLineWidth(1.5)
    c.roundRect(x, y, w, h, 12, fill=1, stroke=1)
    c.setFillColor(text_color)
    c.setFont("Helvetica-Bold", 15)
    lines = label.split("\n")
    line_height = 18
    total = len(lines) * line_height
    y_text = y + h / 2 + total / 2 - line_height
    for line in lines:
        c.drawCentredString(x + w / 2, y_text, line)
        y_text -= line_height


def draw_arrow(c: canvas.Canvas, x1: float, y1: float, x2: float, y2: float, color=ACCENT, label: str = ""):
    c.setStrokeColor(color)
    c.setFillColor(color)
    c.setLineWidth(2)
    c.line(x1, y1, x2, y2)
    angle = 0.35
    arrow_len = 10
    import math

    theta = math.atan2(y2 - y1, x2 - x1)
    for delta in (angle, -angle):
        x = x2 - arrow_len * math.cos(theta + delta)
        y = y2 - arrow_len * math.sin(theta + delta)
        c.line(x2, y2, x, y)
    if label:
        c.setFont("Helvetica", 12)
        c.setFillColor(color)
        c.drawCentredString((x1 + x2) / 2, (y1 + y2) / 2 + 10, label)


def draw_topology(c: canvas.Canvas, title: str):
    draw_title(c, title)
    draw_box(c, 0.9 * inch, 4.5 * inch, 1.6 * inch, 0.8 * inch, "Admin User\nalice", ACCENT)
    draw_box(c, 3.1 * inch, 4.5 * inch, 1.9 * inch, 0.8 * inch, "Browser", ACCENT_2, colors.white)
    draw_box(c, 3.1 * inch, 2.7 * inch, 2.1 * inch, 0.95 * inch, "fl_admin.sh\nhost console", ACCENT_3, colors.white)
    draw_box(c, 6.0 * inch, 3.55 * inch, 2.2 * inch, 1.0 * inch, "Keycloak\nOIDC issuer", colors.HexColor("#EEDFB5"))
    draw_box(c, 9.0 * inch, 3.55 * inch, 2.4 * inch, 1.0 * inch, "NVFlare Server\nserver.admin :8003", colors.HexColor("#E1ECE3"))
    draw_box(c, 9.0 * inch, 1.8 * inch, 1.8 * inch, 0.9 * inch, "Client\nsite-1", colors.HexColor("#F3E4DA"))
    draw_box(c, 11.1 * inch, 1.8 * inch, 1.8 * inch, 0.9 * inch, "Client\nsite-2", colors.HexColor("#F3E4DA"))
    draw_arrow(c, 2.5 * inch, 4.9 * inch, 3.1 * inch, 4.9 * inch)
    draw_arrow(c, 2.1 * inch, 4.5 * inch, 3.7 * inch, 3.65 * inch, label="start CLI")
    draw_arrow(c, 5.0 * inch, 4.9 * inch, 6.0 * inch, 4.05 * inch, label="OIDC")
    draw_arrow(c, 5.2 * inch, 3.15 * inch, 8.95 * inch, 4.05 * inch, label="token login")
    draw_arrow(c, 10.2 * inch, 3.55 * inch, 9.95 * inch, 2.7 * inch, label="jobs")
    draw_arrow(c, 10.7 * inch, 3.55 * inch, 11.95 * inch, 2.7 * inch, label="jobs")
    draw_bullets(
        c,
        [
            "",
            "Keycloak is the human IdP for the demo.",
            "The admin CLI runs on the host.",
            "The server and both sites run in containers.",
        ],
        y_start=1.1 * inch,
        font_size=15,
    )


def draw_trust_split(c: canvas.Canvas, title: str):
    draw_title(c, title)
    c.setFillColor(colors.HexColor("#F4EBDD"))
    c.setStrokeColor(colors.HexColor("#D7C8B0"))
    c.roundRect(0.9 * inch, 2.2 * inch, 4.1 * inch, 3.0 * inch, 16, fill=1, stroke=1)
    c.roundRect(5.5 * inch, 1.7 * inch, 6.9 * inch, 3.5 * inch, 16, fill=1, stroke=1)
    c.setFillColor(TITLE)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(1.15 * inch, 4.85 * inch, "Human Identity Plane")
    c.drawString(5.75 * inch, 4.85 * inch, "Infrastructure Trust Plane")
    draw_box(c, 1.3 * inch, 3.35 * inch, 1.6 * inch, 0.9 * inch, "OIDC issuer", ACCENT_2, colors.white)
    draw_box(c, 3.15 * inch, 3.35 * inch, 1.4 * inch, 0.9 * inch, "Admin CLI", ACCENT_3, colors.white)
    draw_arrow(c, 2.95 * inch, 3.8 * inch, 3.15 * inch, 3.8 * inch)
    draw_box(c, 6.0 * inch, 3.25 * inch, 1.6 * inch, 0.9 * inch, "Server", colors.HexColor("#E1ECE3"))
    draw_box(c, 8.4 * inch, 3.25 * inch, 1.6 * inch, 0.9 * inch, "Client 1", colors.HexColor("#F3E4DA"))
    draw_box(c, 10.8 * inch, 3.25 * inch, 1.6 * inch, 0.9 * inch, "Client 2", colors.HexColor("#F3E4DA"))
    draw_arrow(c, 7.6 * inch, 3.7 * inch, 8.4 * inch, 3.7 * inch, label="mTLS")
    draw_arrow(c, 7.6 * inch, 3.55 * inch, 10.8 * inch, 3.55 * inch, label="mTLS")
    draw_arrow(c, 4.55 * inch, 3.8 * inch, 5.95 * inch, 3.8 * inch, color=ACCENT, label="TLS + bearer token")
    draw_bullets(
        c,
        [
            "",
            "Humans no longer authenticate with project-issued client certs.",
            "Sites still use the existing PKI and mTLS path.",
        ],
        y_start=1.1 * inch,
        font_size=16,
    )


def draw_sequence(c: canvas.Canvas, title: str, participants: List[str], arrows: List[tuple[str, str, str]]):
    draw_title(c, title)
    x_positions = [1.2 * inch, 3.4 * inch, 5.6 * inch, 7.8 * inch, 10.3 * inch]
    top_y = 4.9 * inch
    bottom_y = 1.4 * inch
    for idx, participant in enumerate(participants):
        x = x_positions[idx]
        draw_box(c, x - 0.7 * inch, 5.15 * inch, 1.4 * inch, 0.65 * inch, participant, colors.HexColor("#EEF3F7"))
        c.setStrokeColor(colors.HexColor("#BBB3A7"))
        c.setDash(3, 3)
        c.line(x, top_y, x, bottom_y)
        c.setDash()
    y = 4.45 * inch
    step = 0.48 * inch
    for src, dst, label in arrows:
        x1 = x_positions[participants.index(src)]
        x2 = x_positions[participants.index(dst)]
        draw_arrow(c, x1, y, x2, y, color=ACCENT_2, label=label)
        y -= step


def draw_job_flow(c: canvas.Canvas, title: str):
    draw_title(c, title)
    boxes = [
        (1.0 * inch, "Human admin", ACCENT),
        (3.2 * inch, "fl_admin", ACCENT_2),
        (5.4 * inch, "NVFlare server", ACCENT_3),
        (7.85 * inch, "accepted package\n+ attestation", colors.HexColor("#EEDFB5")),
        (10.55 * inch, "FL clients", colors.HexColor("#E1ECE3")),
    ]
    y = 3.6 * inch
    for x, label, color in boxes:
        draw_box(c, x, y, 1.75 * inch, 1.0 * inch, label, color, colors.white if color != colors.HexColor("#EEDFB5") else TITLE)
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + 1.75 * inch
        x2 = boxes[i + 1][0]
        draw_arrow(c, x1, y + 0.5 * inch, x2, y + 0.5 * inch)
    draw_bullets(
        c,
        [
            "submitter identity comes from OIDC",
            "server hashes and attests the accepted package",
            "clients verify attestation before execution",
        ],
        y_start=1.8 * inch,
        font_size=16,
    )


def draw_generic_slide(c: canvas.Canvas, slide: Slide):
    draw_title(c, slide.title)
    draw_bullets(c, slide.lines)


def draw_slide(c: canvas.Canvas, slide: Slide, slide_num: int):
    draw_background(c, slide_num)
    if slide.title == "What Changes":
        draw_table_slide(c, slide)
    elif slide.title == "Demo Topology":
        draw_topology(c, slide.title)
    elif slide.title == "Trust Split":
        draw_trust_split(c, slide.title)
    elif slide.title == "Browser Login Flow":
        draw_sequence(
            c,
            slide.title,
            ["User", "fl_admin", "Browser", "Keycloak", "Server"],
            [
                ("User", "fl_admin", "start fl_admin.sh"),
                ("fl_admin", "Browser", "open auth URL"),
                ("Browser", "Keycloak", "authenticate"),
                ("Keycloak", "Browser", "auth code"),
                ("Browser", "fl_admin", "localhost callback"),
                ("fl_admin", "Keycloak", "code -> token"),
                ("fl_admin", "Server", "TOKEN_LOGIN"),
                ("Server", "fl_admin", "admin session"),
            ],
        )
    elif slide.title == "Job Submission Integrity":
        draw_job_flow(c, slide.title)
    else:
        draw_generic_slide(c, slide)


def export_pdf(md_path: Path, pdf_path: Path):
    slides = parse_slides(md_path)
    c = canvas.Canvas(str(pdf_path), pagesize=(PAGE_WIDTH, PAGE_HEIGHT))
    c.setTitle("Federated Auth Demo")
    for idx, slide in enumerate(slides, start=1):
        draw_slide(c, slide, idx)
        c.showPage()
    c.save()


def main():
    md_path = Path(__file__).with_name("fedauth_demo_slides.md")
    pdf_path = md_path.with_suffix(".pdf")
    export_pdf(md_path, pdf_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
