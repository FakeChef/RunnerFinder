"""Draws the RunnerFinder application icon (Phase 8.2): a race bib under a magnifying glass.

    <python with Pillow> desktop\\resources\\make_icon.py
writes runnerfinder.png (256 px), runnerfinder.ico (16-256 px) and runnerfinder_1024.png next to this file.
Drawn at 1024 px and scaled down, so it stays crisp at 16 px.
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
S = 1024


def font(size):
    for name in ("segoeuib.ttf", "arialbd.ttf"):
        try:
            return ImageFont.truetype(f"C:/Windows/Fonts/{name}", size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw() -> Image.Image:
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    # tile with a vertical gradient (deep blue -> blue)
    grad = Image.new("RGBA", (S, S))
    gd = ImageDraw.Draw(grad)
    for y in range(S):
        t = y / (S - 1)
        gd.line([(0, y), (S, y)], fill=(int(24 + 22 * t), int(64 + 60 * t), int(150 + 85 * t), 255))
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle((32, 32, S - 32, S - 32), radius=210, fill=255)
    im.paste(grad, (0, 0), mask)
    d = ImageDraw.Draw(im)

    # race bib: white card, magenta number band, two safety-pin holes
    bx0, by0, bx1, by1 = 130, 200, 800, 650
    d.rounded_rectangle((bx0 + 14, by0 + 22, bx1 + 14, by1 + 22), radius=56, fill=(8, 28, 80, 110))   # shadow
    d.rounded_rectangle((bx0, by0, bx1, by1), radius=56, fill=(255, 255, 255, 255))
    d.rounded_rectangle((bx0 + 46, by0 + 120, bx1 - 46, by1 - 70), radius=30, fill=(222, 38, 120, 255))
    for cx in (bx0 + 70, bx1 - 70):
        d.ellipse((cx - 22, by0 + 38, cx + 22, by0 + 82), fill=(24, 64, 150, 255))
    f = font(250)
    d.text(((bx0 + bx1) // 2, (by0 + 120 + by1 - 70) // 2 + 6), "123", font=f, fill=(255, 255, 255, 255),
           anchor="mm")

    # magnifying glass (lower right): translucent lens, the number stays visible through it
    cx, cy, r = 690, 650, 150
    glass = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    g = ImageDraw.Draw(glass)
    g.line((cx + 100, cy + 100, cx + 225, cy + 225), fill=(255, 196, 0, 255), width=86)    # handle
    g.ellipse((cx - r - 30, cy - r - 30, cx + r + 30, cy + r + 30), fill=(255, 196, 0, 255))
    g.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(0, 0, 0, 0))
    g.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(210, 232, 255, 70))
    ring_hole = Image.new("L", (S, S), 0)
    ImageDraw.Draw(ring_hole).ellipse((cx - r, cy - r, cx + r, cy + r), fill=255)
    lens = Image.new("RGBA", (S, S), (210, 232, 255, 70))
    im.alpha_composite(glass)
    tint = Image.composite(lens, Image.new("RGBA", (S, S), (0, 0, 0, 0)), ring_hole)
    im.alpha_composite(tint)
    d = ImageDraw.Draw(im)
    d.arc((cx - r + 26, cy - r + 26, cx + r - 26, cy + r - 26), 200, 250, fill=(255, 255, 255, 230), width=22)  # glint
    return im


def main():
    big = draw()
    big.save(HERE / "runnerfinder_1024.png")
    big.resize((256, 256), Image.LANCZOS).save(HERE / "runnerfinder.png")
    sizes = [16, 20, 24, 32, 40, 48, 64, 128, 256]
    frames = [big.resize((s, s), Image.LANCZOS) for s in sizes]
    frames[-1].save(HERE / "runnerfinder.ico", sizes=[(s, s) for s in sizes], append_images=frames[:-1])
    print("icon written:", HERE)


if __name__ == "__main__":
    main()
