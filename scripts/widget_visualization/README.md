# Widget Visualization

Visual debugging and development technique for PySide6/Qt applications. Capture screenshots at key moments and verify them visually.

## The Technique

Instead of relying on logs or assertions, capture screenshots at each step and have Claude verify them visually. This creates a fast iteration loop for GUI development:

1. Make code change
2. Run script that exercises UI and captures screenshots
3. Claude reviews screenshots (via Haiku delegation)
4. Claude makes next code change
5. Repeat

**Key insight:** The script IS the spec — visual TDD.

## When to Use

- Display corruption (wrong colors, rotation, shearing)
- Widget/component rendering problems
- GUI feature development needing visual validation
- Any bug where "you need to see the actual output"

## Directory Structure

```
scripts/widget_visualization/
├── utils.py              # Core utilities
├── wv_charuco_widget.py  # Widget interaction test
├── wv_capture_volume.py  # 3D OpenGL rendering test
├── wv_full_workflow.py   # Full app smoke test
├── README.md
└── output/               # Screenshots saved here (gitignored)
```

## Running Scripts

```bash
# With display
python scripts/widget_visualization/wv_charuco_widget.py

# Headless (Linux) - required for CI or remote sessions
xvfb-run --auto-servernum python scripts/widget_visualization/wv_full_workflow.py
```

## Core Utilities

```python
from utils import capture_widget, process_events_for, clear_output_dir

# Capture widget to PNG
capture_widget(widget, "01_initial.png")

# Let Qt process events (essential before captures)
process_events_for(500)  # milliseconds

# Clear output directory at start of test
clear_output_dir()
```

## Writing New Scripts

### Naming Convention

Use `wv_` prefix to avoid pytest collection:
- `wv_my_feature.py` (not `test_my_feature.py`)

### Basic Pattern

```python
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from PySide6.QtWidgets import QApplication
from utils import capture_widget, clear_output_dir, process_events_for

def main():
    clear_output_dir()
    app = QApplication(sys.argv)

    # Create your widget
    widget = YourWidget()
    widget.show()
    process_events_for(500)

    # Capture initial state
    capture_widget(widget, "01_initial.png")

    # Interact with widget
    widget.some_button.click()
    process_events_for(300)
    capture_widget(widget, "02_after_click.png")

    # Print verification checklist
    print("Verification checklist:")
    print("  [ ] 01: Widget renders correctly")
    print("  [ ] 02: Button click had expected effect")

if __name__ == "__main__":
    main()
```

### Async Loading Pattern

For widgets that load data asynchronously (like MainWindow loading a workspace):

```python
class WorkflowTest:
    def __init__(self, window):
        self.window = window

    def run(self):
        # launch_workspace returns TaskHandle for connecting callbacks
        handle = self.window.launch_workspace(path)
        handle.completed.connect(self.on_loaded)

    def on_loaded(self):
        # Now safe to interact
        QTimer.singleShot(500, self.capture_state)
```

## Visual Verification with Haiku

After capturing screenshots, spawn a Haiku agent with targeted questions:

```
Use Task tool with model="haiku":

"Review the screenshots in scripts/widget_visualization/output/

We tested [describe what the script does]:
1. Screenshot 01: [what it should show]
2. Screenshot 02: [what it should show]

Verify:
- Does 01 show [specific expected element]?
- Does 02 show [specific change from interaction]?
- Any error dialogs or rendering issues?"
```

The orchestrator provides context about what the script did — Haiku answers specific verification questions rather than generic "describe these images".

### Override: Direct Review

If Haiku's descriptions aren't helping after 2-3 iterations, read the screenshots directly to get unstuck. Note: each screenshot costs ~1.3k tokens in the main context.

## Example Scripts

| Script | Tests | Complexity |
|--------|-------|------------|
| `wv_charuco_widget.py` | Spinbox/checkbox interactions | Simple - direct widget |
| `wv_capture_volume.py` | 3D OpenGL rendering | Medium - data loading |
| `wv_playback_pyvista.py` | PyVista 3D playback, embedding, SVG icons | Medium - data loading |
| `wv_full_workflow.py` | Full app navigation | Complex - async loading |

## Tips

- **OpenGL widgets** need longer delays (1000-1500ms) for initialization
- **Print progress** so you can see where scripts fail
- **Print verification checklist** at the end for quick reference
- **Clear output dir** at start to avoid stale screenshots
- **Use process_events_for()** before every capture — Qt needs time to render

## PyVista/VTK Widget Testing

PyVista widgets have unique challenges due to VTK's OpenGL requirements.

### Headless Mode Limitations

**xvfb + PyVista often segfaults** even with `LIBGL_ALWAYS_SOFTWARE=1`. The VTK rendering pipeline doesn't play well with virtual framebuffers.

**Workaround:** Run with an actual display when testing PyVista widgets:
```bash
# With display (preferred for PyVista)
uv run python scripts/widget_visualization/wv_playback_pyvista.py

# Headless (often crashes with PyVista)
xvfb-run --auto-servernum uv run python scripts/widget_visualization/wv_playback_pyvista.py
```

### Black 3D Views in Software Mode

Even when scripts don't crash, **software rendering may produce black 3D views**. This is a VTK limitation, not a bug in your widget. The Qt controls layer still renders correctly.

**What you CAN verify in software mode:**
- Widget embeds in layouts (no nested window behavior)
- Controls render and respond to clicks
- Icon loading works
- Slider/button state changes

**What you CANNOT reliably verify:**
- 3D scene content (cameras, points, meshes)
- PyVista rendering quality

### Embedding Test Pattern

To verify a `QMainWindow` → `QWidget` refactor worked, embed the widget below a colored header:

```python
container = QMainWindow()
central = QWidget()
layout = QVBoxLayout(central)

# Colored header proves embedding works
header = QLabel("This label is ABOVE the embedded widget")
header.setStyleSheet("background-color: #2196F3; color: white; padding: 10px;")
layout.addWidget(header)

# Your refactored widget
widget = YourPyVistaWidget(view_model)
layout.addWidget(widget, stretch=1)

container.setCentralWidget(central)
```

If the header appears **above** the 3D view (not in a separate window), the refactor succeeded.
