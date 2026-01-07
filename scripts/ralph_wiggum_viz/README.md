# Ralph Wiggum GUI Visual Testing

Visual debugging and development technique for PySide6/Qt applications. Named for the "I'm in danger" meme — when you're deep in visual bugs and need to see what's actually happening.

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
scripts/ralph_wiggum_viz/
├── utils.py              # Core utilities
├── rw_charuco_widget.py  # Widget interaction test
├── rw_capture_volume.py  # 3D OpenGL rendering test
├── rw_full_workflow.py   # Full app smoke test
├── README.md
└── output/               # Screenshots saved here (gitignored)
```

## Running Scripts

```bash
# With display
python scripts/ralph_wiggum_viz/rw_charuco_widget.py

# Headless (Linux) - required for CI or remote sessions
xvfb-run --auto-servernum python scripts/ralph_wiggum_viz/rw_full_workflow.py
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

Use `rw_` prefix to avoid pytest collection:
- `rw_my_feature.py` (not `test_my_feature.py`)

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

"Review the screenshots in scripts/ralph_wiggum_viz/output/

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
| `rw_charuco_widget.py` | Spinbox/checkbox interactions | Simple - direct widget |
| `rw_capture_volume.py` | 3D OpenGL rendering | Medium - data loading |
| `rw_full_workflow.py` | Full app navigation | Complex - async loading |

## Tips

- **OpenGL widgets** need longer delays (1000-1500ms) for initialization
- **Print progress** so you can see where scripts fail
- **Print verification checklist** at the end for quick reference
- **Clear output dir** at start to avoid stale screenshots
- **Use process_events_for()** before every capture — Qt needs time to render
