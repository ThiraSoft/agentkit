# Regenerates the logo files from JetBrains Mono (github.com/JetBrains/JetBrainsMono).
# uv run --with fonttools python docs/logo/build.py <dir with JetBrainsMono-*.ttf> docs/logo
# then rsvg-convert -w <width> <name>.svg -o <name>.png
import sys, os
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen

FONTDIR, OUT = sys.argv[1], sys.argv[2]
CYAN, INK, WHITE = "#00ADD8", "#0B1F2A", "#FFFFFF"
fonts = {w: TTFont(os.path.join(FONTDIR, f"JetBrainsMono-{w}.ttf")) for w in ("ExtraBold", "Medium", "Regular")}

def text(s, x, y, size, weight, color, spacing=0.0):
    f = fonts[weight]; gs = f.getGlyphSet(); cmap = f.getBestCmap(); upm = f["head"].unitsPerEm
    k = size / upm; out = []; cx = x
    for ch in s:
        g = cmap[ord(ch)]
        pen = SVGPathPen(gs)
        gs[g].draw(TransformPen(pen, (k, 0, 0, -k, cx, y)))
        d = pen.getCommands()
        if d: out.append(d)
        cx += gs[g].width * k + spacing
    return f'<path fill="{color}" d="{" ".join(out)}"/>', cx

def mark(x, y, s, mid, color=CYAN):
    ring = ('<path d="M66 22.3 A32 32 0 1 1 34 22.3" fill="none" stroke="{c}" stroke-width="7" stroke-linecap="round"/>'
            '<path d="M41.5 18 L29 15.5 L35.5 29.5 Z" fill="{c}" stroke="{c}" stroke-width="2" stroke-linejoin="round"/>'
            '<circle cx="77.7" cy="66" r="9" fill="{c}"/><circle cx="22.3" cy="66" r="9" fill="{c}"/><circle cx="50" cy="82" r="9" fill="{c}"/>'
            '<circle cx="50" cy="51" r="17" fill="{c}"/>').format(c=color)
    holes = ('<circle cx="77.7" cy="66" r="5"/><circle cx="22.3" cy="66" r="5"/><circle cx="50" cy="82" r="5"/>'
             '<circle cx="44" cy="46.5" r="2.8"/><circle cx="56" cy="46.5" r="2.8"/>'
             '<path d="M42.5 55 Q50 62 57.5 55" fill="none" stroke="#000" stroke-width="3.6" stroke-linecap="round"/>')
    return (f'<g transform="translate({x} {y}) scale({s/100})">'
            f'<mask id="{mid}" maskUnits="userSpaceOnUse" x="-10" y="-10" width="120" height="120">'
            f'<rect x="-10" y="-10" width="120" height="120" fill="#fff"/><g fill="#000">{holes}</g></mask>'
            f'<g mask="url(#{mid})">{ring}</g></g>')

def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img">'
            f'<title>{title}</title>{body}</svg>\n')

def save(name, content):
    open(os.path.join(OUT, name), "w").write(content)

# 1. mark, transparent
save("mark.svg", svg(100, 100, mark(0, 0, 100, "m"), "agentkit"))

# 2. avatar 400x400
save("avatar.svg", svg(400, 400, f'<rect width="400" height="400" fill="{INK}"/>' + mark(70, 66, 260, "m"), "agentkit"))

# 3. README banner 1280x320
def wordmark(x, y, size, agent_color):
    a, nx = text("agent", x, y, size, "ExtraBold", agent_color, spacing=-size*0.02)
    b, ex = text("kit", nx, y, size, "ExtraBold", CYAN, spacing=-size*0.02)
    return a + b, ex
wm, _ = wordmark(330, 168, 104, WHITE)
t1, _ = text("Build LLM agents in Go.", 336, 228, 30, "Medium", "#C9D6DC")
t2, _ = text("tools · MCP · streaming · no magic", 336, 268, 22, "Regular", CYAN)
save("banner.svg", svg(1280, 320,
    f'<rect width="1280" height="320" rx="24" fill="{INK}"/>' + mark(80, 60, 200, "m") + wm + t1 + t2,
    "agentkit: build LLM agents in Go"))

# 4. LinkedIn cover 1584x396
W, H = 1584, 396
body = f'<rect width="{W}" height="{H}" fill="{INK}"/>'
body += f'<g opacity="0.07">{mark(-120, -60, 520, "bg", WHITE)}</g>'
body += mark(470, 62, 130, "m")
wm, _ = wordmark(618, 150, 78, WHITE)
body += wm
body += text("Build LLM agents in Go.", 476, 250, 30, "Medium", "#C9D6DC")[0]
body += text("tools · MCP · streaming · no magic", 476, 296, 21, "Regular", CYAN)[0]
cx, cy, cw, ch = 1040, 58, 492, 280
body += f'<rect x="{cx}" y="{cy}" width="{cw}" height="{ch}" rx="16" fill="#112C3B" stroke="#1E4253"/>'
body += text("main.go", cx + 26, cy + 36, 14, "Regular", "#6F8A97")[0]
P, I, S_, K, C = "#7F96A3", "#D6E4EA", "#A5E07D", CYAN, "#F2C66D"
lines = [
    [("agent, _ := ", I), ("agentkit", K), (".New(ctx, ", I), ("agentkit", K), (".Config{", I)],
    [("    Provider: ", I), ('"anthropic"', S_), (",", P)],
    [("    Model:    ", I), ('"claude-sonnet-5-5"', S_), (",", P)],
    [("    Tools:    ", I), ("[]", P), ("agentkit", K), (".Tool{clock},", I)],
    [("})", P)],
    [("conv := agent.", I), ("NewConversation", C), ("(", P), ('"Be brief."', S_), (")", P)],
    [("turn, _ := conv.", I), ("Send", C), ("(ctx, ", I), ('"What time is it?"', S_), (", h)", I)],
]
y = cy + 82
for ln in lines:
    x = cx + 26
    for s, col in ln:
        p, x = text(s, x, y, 15, "Regular", col)
        body += p
    y += 28
save("linkedin-cover.svg", svg(W, H, body, "agentkit"))
print("ok")

# 5. thumbnail for LinkedIn "Featured" and link previews, 1200x627 (1.91:1)
W, H = 1200, 627
t = f'<rect width="{W}" height="{H}" fill="{INK}"/>'
t += f'<g opacity="0.06">{mark(820, 300, 520, "bg", WHITE)}</g>'
t += mark(200, 178, 230, "m")
wm, _ = wordmark(460, 300, 112, WHITE)
t += wm
t += text("Build LLM agents in Go.", 466, 372, 34, "Medium", "#C9D6DC")[0]
t += text("tools · MCP · streaming", 466, 424, 24, "Regular", CYAN)[0]
save("thumbnail.svg", svg(W, H, t, "agentkit: build LLM agents in Go"))
print("thumbnail ok")
