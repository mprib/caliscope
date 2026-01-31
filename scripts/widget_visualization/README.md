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
│  1. Create/update wv_*_cosmetic.py script for the tab           │
│  2. Spawn UX agent → runs script, views screenshots, assesses   │
│  3. Synthesize assessment → prioritize issues                   │
│  4. Spawn UX agent → implements fixes (one tab at a time)       │
│  5. View screenshots, get user feedback                         │
│  6. Iterate until elegant, then commit                          │
└─────────────────────────────────────────────────────────────────┘
```

**Why this pattern?**
- Orchestrator context stays lean (delegates image viewing to agents)
- UX agent examines screenshots and implements changes
- User provides feedback on screenshots shown by orchestrator
- Each agent is disposable; the workflow persists

**Key practices:**
- **One tab at a time** — Don't modify multiple tabs in parallel
- **UX agent does implementation** — Orchestrator reviews code, preserves context
- **Resume agents** when iterating on the same tab (see Resume Workflow below)
- **User inspects screenshots** — Orchestrator shows images, user gives feedback
- **Commit after each tab** — Small, focused commits

## Resume Workflow for Iterative Polish

When doing cosmetic work, **resume the same UX agent** for follow-up iterations instead of spawning fresh agents. This preserves context and improves efficiency.

### Why Resume?

| Fresh Spawn | Resume |
|-------------|--------|
| Re-reads all files from scratch | Already has files in context |
| No memory of previous attempts | Remembers what was tried |
| Must re-establish style decisions | Carries forward design rationale |
| Higher token cost | Lower incremental cost |
| Good for: new tab, major pivot | Good for: tweaks, refinements |

### When to Resume vs. Fresh Spawn

**Resume the agent when:**
- Making small tweaks based on user feedback ("make that button smaller")
- Iterating on the same component or tab
- Following up on assessment findings
- The agent just completed and you need adjustments

**Fresh spawn when:**
- Moving to a different tab
- Agent context is stale (hours later, different session)
- Major architectural pivot requiring fresh perspective
- Agent got confused or stuck in a bad state

### How to Resume

The Task tool returns an `agentId` after each invocation. Use it with the `resume` parameter:

```
# First invocation (assessment)
Task(subagent_type="pyside6-ui-ux", prompt="Assess the Multi-Camera tab...")
→ Returns agentId: "abc1234"

# Resume for implementation
Task(subagent_type="pyside6-ui-ux", resume="abc1234", prompt="Now implement the button styling fix you identified...")

# Resume again for tweaks
Task(subagent_type="pyside6-ui-ux", resume="abc1234", prompt="User says the button is too wide. Reduce padding.")
```

### Resume Prompt Tips

When resuming, you can write shorter prompts that reference earlier context:

```
# BAD (unnecessary repetition)
"In the Multi-Camera tab widget at src/caliscope/gui/views/multi_camera_processing_widget.py,
the Start Processing button needs the primary button stylesheet from the README..."

# GOOD (leverages preserved context)
"Apply the primary button style to the Start Processing button as discussed."
```

The agent remembers:
- Which files it read
- What issues it identified
- What changes it already made
- The style patterns being followed

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
Primary action   - Filled background (#0078d4), bold text
Secondary action - Ghost (outline only)
Tertiary action  - Text only
```

### Primary Action Button (Reference)

Use this exact stylesheet for primary action buttons (Calibrate, Optimize, etc.):

```python
button.setStyleSheet("""
    QPushButton {
        background-color: #0078d4;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 6px 16px;
        font-weight: bold;
    }
    QPushButton:hover {
        background-color: #106ebe;
    }
    QPushButton:pressed {
        background-color: #005a9e;
    }
    QPushButton:disabled {
        background-color: #555;
        color: #888;
    }
""")
```

### Slider Thumb Styling

Enlarge slider thumbs for easier interaction:

```python
slider.setStyleSheet("""
    QSlider::groove:horizontal {
        height: 8px;
        background: #3a3a3a;
        border-radius: 4px;
    }
    QSlider::handle:horizontal {
        width: 20px;
        height: 20px;
        margin: -6px 0;
        background: #0078d4;
        border-radius: 10px;
    }
    QSlider::handle:horizontal:hover {
        background: #106ebe;
    }
""")
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

### Control Grouping Pattern

Bundle related controls tightly, then center the group:

```python
# Create container for tight grouping
container = QWidget()
container_layout = QVBoxLayout(container)
container_layout.setContentsMargins(0, 0, 0, 0)
container_layout.setSpacing(6)  # Tight internal spacing

container_layout.addWidget(video_frame)
container_layout.addWidget(slider_row)
container_layout.addWidget(button_row)

# Center the container with stretch above/below
parent_layout.addStretch(1)
parent_layout.addWidget(container)
parent_layout.addStretch(1)
```

For horizontal centering of button groups:
```python
row = QHBoxLayout()
row.addStretch()           # Push to center
row.addWidget(button)
row.addWidget(checkbox)
row.addStretch()           # Balance
```

### Internal Stretch Pattern

For forms where some controls should stay grouped while others separate:

```python
# Top controls (stay together)
layout.addWidget(control_1)
layout.addSpacing(12)
layout.addWidget(control_2)
layout.addSpacing(12)
layout.addWidget(control_3)

# Flexible stretch (absorbs extra space)
layout.addStretch()

# Bottom controls (stay together, anchored at bottom)
layout.addWidget(control_4)
layout.addWidget(helper_text)  # No stretch between these
```

### Color Key Legend

When using color-coded status indicators, include a legend:

```python
legend = QLabel(
    '<span style="color: #4CAF50;">●</span> Complete · '
    '<span style="color: #FFA000;">●</span> In Progress · '
    '<span style="color: #2196F3;">●</span> Available · '
    '<span style="color: #666666;">●</span> Not Started'
)
legend.setStyleSheet("color: #888; font-size: 11px;")
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

## Cold-Start Workflow: Polishing a New Tab

When starting cosmetic polish on a tab you haven't touched yet:

### Phase 1: Assessment

1. **Create or update visualization script** (`wv_<tab>_cosmetic.py`)
   - Capture at 3 sizes (900x600, 1200x800, 1600x1000)
   - Capture any relevant states (selection changes, sidebar resize, etc.)

2. **Spawn UX agent for assessment** (review-only, no changes)
   ```
   Prompt: "Run the visualization script and provide a UX assessment.
   Do NOT implement changes — just report findings.

   Evaluate: visual hierarchy, spacing, status indicators,
   responsive behavior, accessibility, control grouping."
   ```

3. **Synthesize into prioritized list**
   - High priority: Accessibility, major layout issues
   - Medium priority: Spacing, visual polish
   - Low priority: Minor refinements

### Phase 2: Implementation

4. **Spawn UX agent for implementation** (one issue category at a time)
   ```
   Prompt: "Implement [specific changes]. Reference the style guide
   in this README. Run visualization script to verify."
   ```

5. **View screenshots, get user feedback**
   - Orchestrator shows images to user
   - User provides specific adjustments

6. **Iterate** — Resume agent or spawn new one for tweaks

7. **Type check** — `uv run basedpyright <modified_files>`

8. **Commit** — Small, focused commit for this tab

### Key Files Pattern

```
wv_<tab>_cosmetic.py     # Visualization script for the tab
<tab>_widget.py          # Main widget file to modify
<tab>_presenter.py       # Usually unchanged (logic layer)
```

### Common First Fixes

Most tabs need these baseline improvements:
- [ ] Button styling (use primary action pattern)
- [ ] Control grouping (tight spacing, centered)
- [ ] Slider thumb enlargement (if sliders present)
- [ ] Status indicators (icon + color, not color-only)
- [ ] Responsive scaling (stretch patterns, size policies)
