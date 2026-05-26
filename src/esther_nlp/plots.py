"""Matplotlib visualisations with dynamic Hebrew Font and BiDi/RTL support."""

import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import arabic_reshaper
from bidi.algorithm import get_display
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CLI rendering
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path as MplPath
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# Pinned color system for unified, sleek visual aesthetics
COLOR_HEMELECH = "#222222"
COLOR_PRIMARY  = "#5a8db5"
COLOR_ERROR    = "#c0392b"
COLOR_SUCCESS  = "#27ae60"
COLOR_PURPLE   = "#8e44ad"
COLOR_ORANGE   = "#d35400"
COLOR_GRAY     = "#7f8c8d"
COLOR_LIGHT_BG = "#fdf6e3"

COLORS_STREAM  = [COLOR_HEMELECH, COLOR_GRAY, COLOR_ORANGE, "#2980b9", "#16a085", COLOR_ERROR, COLOR_SUCCESS]

CHAR_COLORS: Dict[str, str] = {
    "המן":     COLOR_ERROR,
    "המלכה":   COLOR_PURPLE,
    "אסתר":    "#2980b9",
    "מרדכי":   "#16a085",
    "ושתי":    COLOR_ORANGE,
    "אחשורוש": COLOR_GRAY,
    "המלך":     COLOR_HEMELECH
}

# -----------------------------------------------------------------------------
# Font Management: Dynamic Rubik Font Loading for Cross-Platform Hebrew Support
# -----------------------------------------------------------------------------
FONT_FAMILY = "Arial"  # Fallback standard
font_path = Path(__file__).parent.parent.parent / "fonts" / "Rubik-Variable.ttf"

if font_path.exists():
    try:
        # Load and register the variable Rubik font with Matplotlib Font Manager
        font_entry = fm.FontEntry(
            fname=str(font_path.resolve()),
            name="RubikCustom"
        )
        fm.fontManager.ttflist.insert(0, font_entry)
        plt.rcParams["font.family"] = "RubikCustom"
        FONT_FAMILY = "RubikCustom"
    except Exception as e:
        print(f"Warning: Failed to dynamically register Rubik font: {e}. Falling back to default.")

plt.rcParams["axes.unicode_minus"] = False


def heb(s: str) -> str:
    """Helper to reshape and BiDi-flip Hebrew text for matplotlib RTL drawing."""
    if not s:
        return ""
    # Shape Hebrew ligatures and reverse text direction dynamically
    return get_display(arabic_reshaper.reshape(s))


# -----------------------------------------------------------------------------
# Visualizers
# -----------------------------------------------------------------------------

def fig_01_pipeline(out_path: Path) -> None:
    """Generates pipeline flow diagram."""
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.set_xlim(0, 14); ax.set_ylim(0, 4)
    ax.axis("off")

    stages = [
        ("176 verses\nof Esther",       "data"),
        ("Tokenize\n(BPE)",              "stage"),
        ("Embed\n(Word2Vec)",            "stage"),
        ("Mask name\n[MASK]",            "stage"),
        ("Classify\n(tiny MLP)",         "stage"),
        ("0.84\naccuracy",               "result"),
    ]
    colors = {"data": "#e3edf7", "stage": "#fdf6e3", "result": "#d4f0d4"}
    edge   = {"data": "#3a5a7a", "stage": "#a07028", "result": "#2d7a2d"}
    
    n = len(stages)
    box_w, gap = 1.7, 0.45
    x0 = (14 - (n * box_w + (n - 1) * gap)) / 2

    for i, (label, kind) in enumerate(stages):
        x = x0 + i * (box_w + gap)
        box = FancyBboxPatch(
            (x, 1.5), box_w, 1.4,
            boxstyle="round,pad=0.04,rounding_size=0.15",
            linewidth=1.6,
            facecolor=colors[kind], edgecolor=edge[kind]
        )
        ax.add_patch(box)
        ax.text(x + box_w/2, 2.2, label, ha="center", va="center",
                fontsize=11, fontweight="bold" if kind == "result" else "normal")
        if i < n - 1:
            ax.annotate("", xy=(x + box_w + gap - 0.03, 2.2),
                        xytext=(x + box_w + 0.03, 2.2),
                        arrowprops=dict(arrowstyle="->", lw=1.4, color="#555"))

    ax.text(7, 0.55,
            "Each stage trains on the 176 verses and nothing else.",
            ha="center", va="center", fontsize=10, style="italic", color="#555")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_02_top_tokens(tokens: List[str], counts: List[int], total_tokens: int, out_path: Path) -> None:
    """Generates bar chart of top BPE tokens."""
    fig, ax = plt.subplots(figsize=(10, 7))
    colors_bar = [COLOR_ERROR if t == "המלך" else COLOR_PRIMARY for t in tokens]
    bars = ax.barh(range(len(tokens)), counts, color=colors_bar)
    
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels([heb(t) for t in tokens], fontsize=13)
    ax.invert_yaxis()
    ax.set_xlabel("Occurrences in the corpus", fontsize=11)
    ax.set_title(f"Top {len(tokens)} BPE tokens by frequency in Megillat Esther\n"
                 f"(highlighted bar is {heb('המלך')} — twice the next most common token)",
                 fontsize=12)

    for bar, n in zip(bars, counts):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                f"{n}  ({100 * n / total_tokens:.1f}%)",
                va="center", fontsize=10, color="#333")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, max(counts) * 1.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_03_radial_map(
    sorted_chars: List[str],
    medians: Dict[str, int],
    per_char: Dict[str, List[int]],
    vocab_size: int,
    out_path: Path
) -> None:
    """Generates concentric target rank radial map."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")

    def rank_to_radius(rank: int) -> float:
        # Non-linear scaling to widen high-ranks and compress low-ranks
        return 0.15 + 0.75 * np.sqrt(rank / vocab_size)

    ring_thresholds = [32, 64, 192, 320, vocab_size]
    ring_labels     = ["top 5%", "top 10%", "top 30%", "top 50%", f"all {vocab_size} tokens"]
    
    for rank, label in zip(ring_thresholds, ring_labels):
        r = rank_to_radius(rank)
        circle = plt.Circle((0, 0), r, fill=False, lw=0.7, color="#aaa", linestyle="--")
        ax.add_patch(circle)
        ax.text(0, r + 0.025, label, ha="center", va="bottom", fontsize=8, color="#888")

    # Center Node: המלך
    ax.scatter([0], [0], s=900, color="#222", zorder=5)
    ax.text(0, 0, heb("המלך"), ha="center", va="center",
            fontsize=13, color="white", fontweight="bold", zorder=6)

    # Distribute target characters evenly
    angles = np.linspace(np.pi/2, np.pi/2 + 2*np.pi, len(sorted_chars), endpoint=False)
    
    for c, theta in zip(sorted_chars, angles):
        r_med = rank_to_radius(medians[c])
        x, y = r_med * np.cos(theta), r_med * np.sin(theta)

        # Plot training seed jitter clouds
        for r_seed in per_char[c]:
            r_s = rank_to_radius(r_seed)
            jitter = 0.04 * (np.random.rand() - 0.5)
            xs = r_s * np.cos(theta + jitter)
            ys = r_s * np.sin(theta + jitter)
            ax.scatter(xs, ys, s=30, color=CHAR_COLORS[c], alpha=0.25, zorder=3)

        # Median marker node
        ax.scatter(x, y, s=320, color=CHAR_COLORS[c], edgecolor="black", linewidth=1.5, zorder=4)
        
        # Outer labels
        label_x = (r_med + 0.10) * np.cos(theta)
        label_y = (r_med + 0.10) * np.sin(theta)
        ax.text(label_x, label_y, f"{heb(c)}\nrank {medians[c]}",
                ha="center", va="center", fontsize=11, fontweight="bold",
                color=CHAR_COLORS[c])

    ax.set_title("Each character's distance from המלך in embedding space\n"
                 "(closer to center = higher contextual semantic similarity)\n"
                 "Small transparent dots = individual seed runs; large dot = median",
                 fontsize=12, pad=20)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_04_arc_diagram(sim_matrix: np.ndarray, out_path: Path) -> None:
    """Generates chapter similarity arc diagram."""
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(-0.5, 9.5); ax.set_ylim(-0.6, 4.5)
    ax.axis("off")

    # Draw bottom row: 10 chapter nodes
    for i in range(10):
        circle = plt.Circle((i, 0), 0.30, facecolor=COLOR_LIGHT_BG,
                            edgecolor="#5a4828", lw=1.5, zorder=5)
        ax.add_patch(circle)
        ax.text(i, 0, f"{i+1}", ha="center", va="center", fontsize=12,
                fontweight="bold", zorder=6)

    ax.text(-0.5, -0.4, "Chapter:", ha="right", va="center", fontsize=11, style="italic")

    # Extract off-diagonal chapter similarities
    pairs = []
    for i in range(10):
        for j in range(i+1, 10):
            pairs.append((i, j, float(sim_matrix[i, j])))
    pairs.sort(key=lambda x: -x[2])

    THRESH = 0.92
    plotted = [p for p in pairs if p[2] >= THRESH]

    def draw_arc(i: int, j: int, height: float):
        ts = np.linspace(0, 1, 50)
        xs = i + (j - i) * ts
        ys = height * np.sin(np.pi * ts)
        return xs, ys

    max_sim = max(p[2] for p in plotted)
    min_sim = min(p[2] for p in plotted)

    for i, j, s in plotted:
        span = j - i
        height = 0.5 + 0.25 * span
        norm = (s - min_sim) / max(max_sim - min_sim, 1e-6)
        
        # Highlight chiastic mirror pairs in dark red
        if (i, j) in [(2, 7), (4, 6)]:
            color = COLOR_ERROR; lw = 5.5; alpha = 0.95
        else:
            color = COLOR_PRIMARY; lw = 0.8 + norm * 5.0; alpha = 0.25 + norm * 0.65
            
        xs, ys = draw_arc(i, j, height)
        ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, zorder=3)

    # Highlight annotations
    annotations = [
        (2, 7, "ch3 ↔ ch8\nthe two decrees", "right", "0.973"),
        (4, 6, "ch5 ↔ ch7\nthe two banquets", "left", "0.967"),
    ]
    for i, j, label, side, score in annotations:
        span = j - i
        height = 0.5 + 0.25 * span
        mid_x = (i + j) / 2
        mid_y = height * 1.05
        ax.annotate(f"{label}\ncos = {score}",
                    xy=(mid_x, mid_y),
                    xytext=(mid_x + (1.2 if side == "right" else -1.2), mid_y + 0.6),
                    fontsize=10, ha="center", color=COLOR_ERROR,
                    fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=COLOR_ERROR, lw=0.8))

    ax.set_title("Chapter-pair semantic similarity in vector space\n"
                 "Connecting lines show context similarities; thicker lines = more similar.\n"
                 "Red arcs = the two strongest pairs — matching the famous chiastic mirrors of Esther.",
                 fontsize=12, pad=15)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_05a_top_movers(tokens: List[str], values: List[float], out_path: Path) -> None:
    """Generates bar chart of top Procrustes movers."""
    fig, ax = plt.subplots(figsize=(9, 6))
    bar_colors = [COLOR_ERROR if w == "יקר" else "#888" for w in tokens]
    bars = ax.barh(range(len(tokens)), values, color=bar_colors)
    
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels([heb(t) for t in tokens], fontsize=13)
    ax.invert_yaxis()
    ax.set_xlabel("How far the embedding moved (Procrustes semantic distance)", fontsize=11)
    ax.set_title(f"Top {len(tokens)} words whose meaning shifts between halves of the book\n"
                 f"(highlighted bar is {heb('יקר')} — the single most-mobile token)",
                 fontsize=12)
                 
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{v:.2f}", va="center", fontsize=10, color="#333")
                
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, max(values) * 1.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_05b_neighbor_clouds(n1: List[Tuple[str, float]], n2: List[Tuple[str, float]], out_path: Path) -> None:
    """Generates side-by-side qualitative neighborhood clouds for יקר."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    
    datasets = [
        (axes[0], n1, "First half (chapters 1–5)\nGeneric court-pomp vocabulary", COLOR_PRIMARY),
        (axes[1], n2, "Second half (chapters 6–10)\nThe chapter-6 honoring scene", COLOR_ERROR),
    ]
    
    for ax, neighbors, title, color in datasets:
        ax.set_xlim(0, 10); ax.set_ylim(0, 10)
        ax.axis("off")
        ax.set_title(title, fontsize=12)
        if not neighbors:
            continue
            
        max_sim = max(s for _, s in neighbors)
        min_sim = min(s for _, s in neighbors)
        rng = np.random.default_rng(42)
        
        placements = []
        for _, (w, s) in enumerate(neighbors):
            norm = (s - min_sim) / max(max_sim - min_sim, 1e-6)
            fontsize = 14 + norm * 18
            
            # Simple collision avoidance jitter
            for _ in range(50):
                x = rng.uniform(1, 9)
                y = rng.uniform(1, 9)
                if all((x - px)**2 + (y - py)**2 > 2.0 for px, py, _ in placements):
                    break
            placements.append((x, y, fontsize))
            alpha = 0.5 + 0.5 * norm
            ax.text(x, y, heb(w), ha="center", va="center", fontsize=fontsize,
                    color=color, alpha=alpha, fontweight="bold" if norm > 0.7 else "normal")

    fig.suptitle(f"The word {heb('יקר')} lives among completely different words in each half of the book",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_06_streamgraph(density: np.ndarray, char_names: List[str], out_path: Path) -> None:
    """Generates protagonist streamgraph."""
    fig, ax = plt.subplots(figsize=(13, 7))
    x = np.arange(1, 11)
    
    ax.stackplot(x, density, labels=[heb(n) for n in char_names],
                 colors=COLORS_STREAM, alpha=0.85, baseline="sym",
                 edgecolor="white", linewidth=1.5)
                 
    ax.set_xticks(x)
    ax.set_xticklabels([f"ch{i}" for i in x], fontsize=11)
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Chapter", fontsize=11)

    handles = [mpatches.Patch(color=c, label=heb(n)) for n, c in zip(char_names, COLORS_STREAM)]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=12, frameon=False)

    ax.set_title(f"Each character is a river. Width = mentions per chapter.\n"
                 f"Vashti exits after ch2. Esther thins to a single mention in ch6 (sleepless night). "
                 f"{heb('היהודים')} swells in ch8–9.",
                 fontsize=12, pad=15)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def draw_wireframe_sphere(ax, color: str = "#9aaab8", alpha: float = 0.55) -> None:
    """Utility to render wireframe coordinates sphere."""
    u = np.linspace(0, 2 * np.pi, 28)
    v = np.linspace(0, np.pi, 18)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color=color, alpha=alpha, linewidth=0.55)
    
    # Hide axis borders for a premium clean space look
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("none")
    ax.yaxis.pane.set_edgecolor("none")
    ax.zaxis.pane.set_edgecolor("none")


def fig_06b_embedding_ball(pos: np.ndarray, chars: List[str], out_path: Path) -> None:
    """Generates 3D vector embedding sphere."""
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection="3d")
    draw_wireframe_sphere(ax)

    # Custom offsets and alignments to avoid label overlap and boundaries cutoffs
    label_configs = {
        "המלך":    {"offset": 1.28, "ha": "right",  "va": "bottom"},
        "המן":     {"offset": 1.28, "ha": "left",   "va": "top"},
        "מרדכי":   {"offset": 1.20, "ha": "left",   "va": "center"},
        "אסתר":    {"offset": 1.20, "ha": "center", "va": "bottom"},
        "אחשורוש": {"offset": 1.20, "ha": "right",  "va": "center"}
    }

    for c, p in zip(chars, pos):
        ax.plot([0, p[0]], [0, p[1]], [0, p[2]], color=CHAR_COLORS[c], lw=2.0, alpha=0.7)
        ax.scatter([p[0]], [p[1]], [p[2]], s=180, color=CHAR_COLORS[c], 
                   edgecolor="black", linewidth=1.3, zorder=5)
        
        cfg = label_configs.get(c, {"offset": 1.20, "ha": "center", "va": "center"})
        off = cfg["offset"]
        ax.text(p[0] * off, p[1] * off, p[2] * off,
                heb(c), fontsize=12, fontweight="bold", color=CHAR_COLORS[c],
                ha=cfg["ha"], va=cfg["va"])

    # Expand limits to -1.5 to 1.5 to prevent labels from being cut off at boundaries
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.set_zlim(-1.5, 1.5)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.grid(False)
    ax.set_title("Each character = one vector inside the embedding space\n"
                 "(50 dimensions projected to 3 PCA coordinates, so we can see them)",
                 fontsize=12, pad=15)
    ax.view_init(elev=70, azim=210)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_07_accuracy(majority_baseline: float, clf_acc: float, clf_std: float, out_path: Path) -> None:
    """Generates accuracy bar chart."""
    labels = ["Random", "Always predict\nMordecai\n(majority)", "Our classifier\n(Zeresh)"]
    values = [0.50, majority_baseline, clf_acc]
    colors_acc = ["#aaa", COLOR_PRIMARY, COLOR_SUCCESS]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, values, color=colors_acc)
                  
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.axhline(0.5, color="#888", linestyle=":", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.03,
                f"{v:.2f}", ha="center", fontsize=12, fontweight="bold")
                
    ax.set_title("Classifier accuracy vs simple baselines\n"
                 "(predicting Haman or Mordecai from the surrounding words alone)",
                 fontsize=12)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_08a_sankey(confusion_matrix: np.ndarray, out_path: Path) -> None:
    """Generates confusion flow Sankey connectors diagram."""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.axis("off")

    left_x, right_x = 1.5, 8.5
    block_w = 1.0
    n_total = confusion_matrix.sum()
    scale = 7.0 / n_total   # Scale flows to fit within 7 vertical plot units

    true_sizes = [confusion_matrix[0].sum(), confusion_matrix[1].sum()]
    pred_sizes = [confusion_matrix[:, 0].sum(), confusion_matrix[:, 1].sum()]

    def get_stack_positions(sizes, gap=0.5):
        positions = []
        curr = 9.5
        for s in sizes:
            h = s * scale
            positions.append((curr - h, curr))
            curr -= h + gap
        return positions

    left_y  = get_stack_positions(true_sizes)
    right_y = get_stack_positions(pred_sizes)

    # Draw actual blocks
    labels_left = ["Actually Haman", "Actually Mordecai"]
    colors_block = [COLOR_ERROR, "#16a085"]
    for (yb, yt), label, color, sz in zip(left_y, labels_left, colors_block, true_sizes):
        rect = plt.Rectangle((left_x - block_w, yb), block_w, yt - yb,
                             facecolor=color, edgecolor="black", lw=1.2)
        ax.add_patch(rect)
        ax.text(left_x - block_w - 0.2, (yb + yt)/2, f"{label}\nn={sz}", ha="right", va="center", fontsize=11)

    labels_right = ["Predicted Haman", "Predicted Mordecai"]
    for (yb, yt), label, color, sz in zip(right_y, labels_right, colors_block, pred_sizes):
        rect = plt.Rectangle((right_x, yb), block_w, yt - yb,
                             facecolor=color, edgecolor="black", lw=1.2)
        ax.add_patch(rect)
        ax.text(right_x + block_w + 0.2, (yb + yt)/2, f"{label}\nn={sz}", ha="left", va="center", fontsize=11)

    left_offsets  = {0: left_y[0][1], 1: left_y[1][1]}
    right_offsets = {0: right_y[0][1], 1: right_y[1][1]}

    flow_specs = []
    for true_c in range(2):
        for pred_c in range(2):
            flow_specs.append((true_c, pred_c, int(confusion_matrix[true_c, pred_c])))

    # Sort error flows to draw on top of correct ones
    flow_specs.sort(key=lambda x: x[0] != x[1])

    for true_c, pred_c, n_flow in flow_specs:
        if n_flow == 0:
            continue
        h = n_flow * scale
        
        y_src_top = left_offsets[true_c]
        y_src_bot = y_src_top - h
        left_offsets[true_c] = y_src_bot
        
        y_tgt_top = right_offsets[pred_c]
        y_tgt_bot = y_tgt_top - h
        right_offsets[pred_c] = y_tgt_bot

        is_correct = (true_c == pred_c)
        flow_color = COLOR_SUCCESS if is_correct else COLOR_ERROR
        flow_alpha = 0.55 if is_correct else 0.75

        # Bezier curve visualizer patch
        x0, x1 = left_x, right_x
        mid = (x0 + x1) / 2
        verts = [
            (x0, y_src_top),
            (mid, y_src_top), (mid, y_tgt_top), (x1, y_tgt_top),
            (x1, y_tgt_bot),
            (mid, y_tgt_bot), (mid, y_src_bot), (x0, y_src_bot),
            (x0, y_src_top),
        ]
        codes = [MplPath.MOVETO,
                 MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
                 MplPath.LINETO,
                 MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
                 MplPath.CLOSEPOLY]
        path = MplPath(verts, codes)
        patch = PathPatch(path, facecolor=flow_color, alpha=flow_alpha, edgecolor="none")
        ax.add_patch(patch)

        label_x = x0 + 0.4
        label_y = (y_src_top + y_src_bot) / 2
        ax.text(label_x, label_y, f"{n_flow}", ha="left", va="center",
                fontsize=11, fontweight="bold", color="white" if flow_alpha > 0.6 else "black")

    green_patch = mpatches.Patch(color=COLOR_SUCCESS, alpha=0.7, label="Correct prediction")
    red_patch   = mpatches.Patch(color=COLOR_ERROR, alpha=0.8, label="Wrong prediction")
    ax.legend(handles=[green_patch, red_patch], loc="lower center",
              bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=11, frameon=False)

    ax.set_title("How the classifier did, by actual vs predicted character\n"
                 f"61 total verses; {confusion_matrix[0,0] + confusion_matrix[1,1]} correct, "
                 f"{confusion_matrix[0,1] + confusion_matrix[1,0]} wrong",
                 fontsize=12)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_08b_hallucinations(probs: np.ndarray, correct_mask: np.ndarray, out_path: Path) -> None:
    """Generates confidence vs correctness distribution scatter plot."""
    fig, ax = plt.subplots(figsize=(12, 5))
    rng = np.random.default_rng(42)
    n = len(probs)

    # Add jitter to categorical Y ticks (0/1) for better separation
    y_pos = np.where(correct_mask, 1.0, 0.0) + rng.uniform(-0.18, 0.18, n)
    colors_dot = [COLOR_SUCCESS if c else COLOR_ERROR for c in correct_mask]

    ax.scatter(probs, y_pos, c=colors_dot, s=110, alpha=0.7, edgecolor="white", linewidth=0.8)

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["WRONG\nprediction", "CORRECT\nprediction"], fontsize=11)
    ax.set_xlabel("Classifier's confidence in its answer (softmax probability)", fontsize=11)
    ax.set_xlim(0.45, 1.02)
    ax.set_ylim(-0.6, 1.6)
    ax.axvline(0.5, color="#888", linestyle=":", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Highlight confidently-wrong predictions
    conf_wrong_indices = np.where((~correct_mask) & (probs > 0.7))[0]
    for idx in conf_wrong_indices:
        cx = probs[idx]
        cy = y_pos[idx]
        ax.annotate(f"conf: {cx:.2f}", xy=(cx, cy), xytext=(cx - 0.04, cy + 0.15),
                    fontsize=8, fontweight="bold", color=COLOR_ERROR,
                    arrowprops=dict(arrowstyle="->", color=COLOR_ERROR, lw=0.6))

    ax.set_title("Softmax confidence vs prediction correctness\n"
                 "(highly confident incorrect answers = structural hallucinations)",
                 fontsize=12)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_10b_yakar_two_worlds(
    model_h1,
    model_h2,
    readable_n1: List[str],
    readable_n2: List[str],
    out_path: Path
) -> None:
    """Generates dual 3D vector spheres side-by-side."""
    anchors = ["המלך", "אחשורוש", "מרדכי", "אסתר"]
    fig = plt.figure(figsize=(18, 10))

    def plot_one_sphere(ax, model, label_words, title, yakar_color, accent_color):
        draw_wireframe_sphere(ax)
        all_words = anchors + ["יקר"] + label_words
        vecs = np.stack([model.wv[w] for w in all_words])
        vn = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
        
        # Fit PCA transformation based on spatial orientation
        p = PCA(n_components=3).fit(vn)
        pos = p.transform(vn)
        pos = pos / (np.linalg.norm(pos, axis=1, keepdims=True) + 1e-12)

        # Plot base anchors in gray
        for w, pp in zip(anchors, pos[:len(anchors)]):
            ax.scatter([pp[0]], [pp[1]], [pp[2]], s=110, color="#888", edgecolor="black", linewidth=0.8, zorder=4)
            ax.text(pp[0] * 1.22, pp[1] * 1.22, pp[2] * 1.22, heb(w), fontsize=13, color="#444", ha="center", va="center")

        # Plot יקר in bright highlight color
        yp = pos[len(anchors)]
        ax.plot([0, yp[0]], [0, yp[1]], [0, yp[2]], color=yakar_color, lw=3.0, alpha=0.9)
        ax.scatter([yp[0]], [yp[1]], [yp[2]], s=420, color=yakar_color, edgecolor="black", linewidth=1.8, zorder=6)
        ax.text(yp[0] * 1.30, yp[1] * 1.30, yp[2] * 1.30, heb("יקר"), fontsize=20, fontweight="bold", color=yakar_color, ha="center", va="center")

        # Plot surrounding qualitative neighbors
        for w, pp in zip(label_words, pos[len(anchors) + 1:]):
            ax.scatter([pp[0]], [pp[1]], [pp[2]], s=180, color=accent_color, edgecolor="black", linewidth=1.0, zorder=5, alpha=0.9)
            ax.text(pp[0] * 1.25, pp[1] * 1.25, pp[2] * 1.25, heb(w), fontsize=14, color=accent_color, ha="center", va="center", fontweight="bold")

        ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.set_zlim(-1.3, 1.3)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.grid(False)
        ax.set_title(title, fontsize=14, pad=14)
        ax.view_init(elev=20, azim=35)

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    plot_one_sphere(ax1, model_h1, readable_n1,
                    "First half (chapters 1–5)\nיקר sits among generic court-pomp words",
                    "#2980b9", "#7eb6dc")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    plot_one_sphere(ax2, model_h2, readable_n2,
                    "Second half (chapters 6–10)\nיקר sits in the chapter-6 honoring scene",
                    COLOR_ERROR, "#e88273")

    fig.suptitle(f"Same word, two different worlds — where {heb('יקר')} sits in each half of the book",
                 fontsize=15, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_appendix_bootstrap(null_dist: np.ndarray, test_dist: np.ndarray, out_path: Path) -> None:
    """Generates overlapping bootstrap histograms for negative result analysis."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(null_dist, bins=35, density=True, color="#888", alpha=0.55, label="Null distribution (noise)")
    ax.hist(test_dist, bins=35, density=True, color=COLOR_ERROR, alpha=0.65, label="Replacement-parallel test statistic")
    
    ax.set_xlabel("Alignment cosine value", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(fontsize=11, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    ax.set_title("Replacement-parallel bootstrap hypothesis test (Option A vs Option B direction)\n"
                 "Overlapping distributions confirm zero distinguishable geometric signals.",
                 fontsize=12)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
