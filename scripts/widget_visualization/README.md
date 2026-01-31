# Widget Visualization

Visual development workflow for PySide6/Qt applications. Combines automated screenshot capture with AI-driven design review to achieve **elegant, professional UI**.

## Philosophy

This isn't just debugging — it's **visual design iteration**. The goal is software that looks crafted, not just functional.

**Design Principles (Tufte-inspired):**
- **Data-ink ratio**: Every pixel should communicate. Remove visual noise.
- **Hierarchy through subtlety**: Use spacing, weight, and value — not heavy borders.
- **Consistent rhythm**: Spacing should follow a scale (4, 8, 16, 24, 32px).
- **Elegance over decoration**: Beauty comes from balance and proportion, not ornament.

**The test**: "Would I proudly demo this?" If not, keep iterating.

## The Orchestrated Workflow

The orchestrator (main Claude session) coordinates specialized agents:

```
┌─────────────────────────────────────────────────────────────────┐
│  ORCHESTRATOR (preserves context, makes decisions)              │
│                                                                 │
│  1. Spawn CODER agent → implements UI changes                   │
│  2. Run visualization script → captures screenshots             │
│  3. Spawn UX agent → reviews screenshots, reports findings      │
│  4. Synthesize feedback → decide next iteration                 │
│  5. Repeat until elegant                                        │
└─────────────────────────────────────────────────────────────────┘
```

**Why this pattern?**
- Orchestrator context stays lean (doesn't load images)
- UX agent can examine screenshots in detail
- Coder agent focuses on implementation
- Each agent is disposable; the workflow persists

## Prompting the UX Agent for Elegance

Don't just ask "does it work?" — prompt for **good taste**:

```
You are reviewing [widget/tab]. Your job is not just to check if things
work — it's to evaluate elegance and good taste.

Design Philosophy:
This should look like professional software — think Blender, DaVinci
Resolve, Figma. Clean, purposeful, refined.

Evaluate with these criteria:
1. Visual Rhythm — Consistent spacing? Intentional placement?
2. Typography — Clear hierarchy? Unnecessary boldness?
3. Whitespace — Purposeful grouping? Or dead space?
4. Color Palette — Harmonious? Semantic meaning clear?
5. The "Would I Show This Off?" Test

What's missing to make this elegant? Be specific.
```

## Multi-Size Capture Pattern

Always test resize behavior. Capture at multiple window sizes:

```python
sizes = [
    (900, 600, "small"),
    (1200, 800, "medium"),
    (1600, 1000, "large"),
]

for width, height, label in sizes:
    window.resize(width, height)
    process_events_for(400)
    capture_widget(window, f"01_widget_{label}.png")
```

This reveals:
- Layout breaking at small sizes
- Awkward whitespace at large sizes
- Elements that don't scale proportionally

## Directory Structure

```
scripts/widget_visualization/
├── utils.py              # Core utilities
├── wv_*.py               # Visualization scripts (disposable)
├── README.md             # This file
└── output/               # Screenshots (gitignored)
```

Scripts are **ephemeral working files**. Create them as needed, delete when done.

## Running Scripts

```bash
# Direct execution
uv run python scripts/widget_visualization/wv_project_tab_cosmetic.py

# Headless (CI or remote)
xvfb-run uv run python scripts/widget_visualization/wv_project_tab_cosmetic.py
```

## Core Utilities

```python
from utils import capture_widget, process_events_for, clear_output_dir

clear_output_dir()                        # Remove old screenshots
process_events_for(500)                   # Let Qt render (ms)
capture_widget(widget, "01_initial.png")  # Save screenshot
```

## Writing Visualization Scripts

### Naming Convention

Use `wv_` prefix to avoid pytest collection:
- `wv_my_feature.py` (not `test_my_feature.py`)

### Basic Pattern

```python
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from PySide6.QtWidgets import QApplication
from utils import capture_widget, clear_output_dir, process_events_for

def main():
    clear_output_dir()
    app = QApplication(sys.argv)

    # Create widget with controlled context
    widget = YourWidget()
    widget.show()

    # Capture at multiple sizes
    for width, height, label in [(900, 600, "sm"), (1200, 800, "md"), (1600, 1000, "lg")]:
        widget.window().resize(width, height)
        process_events_for(400)
        capture_widget(widget.window(), f"01_{label}.png")

    print("Screenshots: scripts/widget_visualization/output/")

if __name__ == "__main__":
    main()
```

## Iteration Checklist

When doing cosmetic polish work:

1. **Structural changes first** — Layout, component hierarchy
2. **Then visual refinement** — Icons, typography, spacing
3. **Then polish** — Hover states, subtle animations, shadows

At each stage:
- [ ] Capture screenshots at 3 sizes
- [ ] UX agent reviews for elegance (not just function)
- [ ] Address highest-impact issues first
- [ ] Re-capture and verify

## Design Quick Reference

### Spacing Scale
```
4px   - Tight (within related items)
8px   - Standard (between items)
16px  - Comfortable (between groups)
24px  - Generous (between sections)
32px  - Dramatic (major divisions)
```

### Typography Hierarchy
```
Section headers  - Semibold, larger
Primary text     - Regular weight
Secondary text   - Regular weight, lighter color (#888)
Helper text      - Smaller, lighter color, optional italic
```

### Status Colors
```
Complete     - #4CAF50 (green)
Incomplete   - #FFA000 (amber)
Available    - #2196F3 (blue)
Not started  - #666666 (gray)
```

### Button Styles
```
Primary action   - Filled background
Secondary action - Ghost (outline only)
Tertiary action  - Text only
```

### Form Layout Standards
```
Row-to-row spacing     - 12px  (breathing room without waste)
Section spacing        - 24px  (clear visual grouping)
Group box padding      - 16px  (consistent container feel)
Label-to-input gap     - 8px   (horizontal, within a row)
```

### List Widget Standards
```
Row height             - 36-40px (touch-friendly, scannable)
Left padding           - 8-12px  (text inset from edge)
Selection highlight    - #3a5f8a (strong, not default system)
```

### Preview/Media Scaling

**Principle:** Fill available space within bounds — never fixed size.

```
min_dimension: 150px   (ensures readability at small windows)
max_dimension: 450px   (prevents comical enlargement)
               — OR 50% of container width, whichever smaller
aspect_ratio:  Always preserved
dead_space:    Fill with subtle background (#1e1e1e), not black void (#000)
```

Implementation pattern:
```python
# BAD: Fixed size that doesn't adapt
max_dimension = 350  # Same at all window sizes

# GOOD: Responsive within bounds
available = min(container.width() * 0.5, container.height() - 40)
target = max(150, min(available, 450))
```

### Accessibility: Status Indicators

**Rule:** Never use color alone to convey status.

```
✓ Icon + color         (checkmark + green, X + red)
✓ Icon + text          ("Complete", "Failed")
✗ Color-only text      (green text vs red text — colorblind users miss this)
```

Recommended pattern for lists:
```
[●] Port 0 — 0.42px    (green dot + RMSE for calibrated)
[○] Port 1             (gray circle for uncalibrated)
```

### Visual Separation

Create zones with subtle cues, not heavy borders:

```
Background tint        - #1e1e1e vs #1a1a1a (barely perceptible)
Separator line         - 1px solid #333 (subtle, not harsh)
Shadow                 - Avoid — too "web 2.0"
Heavy borders          - Avoid — creates visual noise
```

## Tips

- **OpenGL widgets** need 1000-1500ms delays for initialization
- **Print progress** so you can see where scripts fail
- **Clear output dir** at start to avoid stale screenshots
- **Use process_events_for()** before every capture
- **Scripts are disposable** — don't precious them

## PyVista/VTK Notes

Software rendering may produce black 3D views (VTK limitation). Qt controls still render correctly.

**Can verify in software mode:**
- Widget embeds properly
- Controls render and respond
- Icon loading works

**Cannot verify reliably:**
- 3D scene content
- Rendering quality
