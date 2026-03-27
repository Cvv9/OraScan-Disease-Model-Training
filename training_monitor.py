"""
Training Monitor GUI - Real-time visualization of model training progress.

Reads live training data from a JSON progress file written by the training
script, and displays interactive charts for loss, accuracy, and per-class
metrics.

Usage:
    python training_monitor.py
    python training_monitor.py --progress-file path/to/training_progress.json
"""

import argparse
import json
import sys
import tkinter as tk
from tkinter import ttk
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Default progress file location
DEFAULT_PROGRESS = Path(r"E:\ECG and Dental Images\Dental data set\Models\training_progress.json")

# Colors
BG_COLOR = "#1e1e2e"
FG_COLOR = "#cdd6f4"
ACCENT = "#89b4fa"
GREEN = "#a6e3a1"
RED = "#f38ba8"
YELLOW = "#f9e2af"
SURFACE = "#313244"
OVERLAY = "#45475a"


class TrainingMonitor:
    def __init__(self, root, progress_file):
        self.root = root
        self.progress_file = Path(progress_file)
        self.last_modified = 0
        self.data = None

        # Window setup
        self.root.title("OraScan Training Monitor")
        self.root.geometry("1200x800")
        self.root.configure(bg=BG_COLOR)
        self.root.minsize(900, 600)

        # Style
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.TFrame", background=BG_COLOR)
        style.configure("Dark.TLabel", background=BG_COLOR, foreground=FG_COLOR,
                         font=("Segoe UI", 10))
        style.configure("Header.TLabel", background=BG_COLOR, foreground=FG_COLOR,
                         font=("Segoe UI", 14, "bold"))
        style.configure("Metric.TLabel", background=SURFACE, foreground=FG_COLOR,
                         font=("Segoe UI", 11))
        style.configure("Value.TLabel", background=SURFACE, foreground=ACCENT,
                         font=("Segoe UI", 20, "bold"))
        style.configure("Status.TLabel", background=BG_COLOR, foreground=YELLOW,
                         font=("Segoe UI", 10))

        self._build_ui()
        self._poll_progress()

    def _build_ui(self):
        # Main container
        main = ttk.Frame(self.root, style="Dark.TFrame")
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ── Header row ──────────────────────────────────────────────
        header_frame = ttk.Frame(main, style="Dark.TFrame")
        header_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(header_frame, text="OraScan Training Monitor",
                  style="Header.TLabel").pack(side=tk.LEFT)

        self.status_label = ttk.Label(header_frame, text="Waiting for data...",
                                       style="Status.TLabel")
        self.status_label.pack(side=tk.RIGHT)

        # ── Metric cards row ────────────────────────────────────────
        cards_frame = ttk.Frame(main, style="Dark.TFrame")
        cards_frame.pack(fill=tk.X, pady=(0, 10))

        self.cards = {}
        card_defs = [
            ("epoch", "Epoch", "0 / 0"),
            ("phase", "Phase", "--"),
            ("train_acc", "Train Acc", "0.0%"),
            ("val_acc", "Val Acc", "0.0%"),
            ("best_val", "Best Val Acc", "0.0%"),
            ("best_epoch", "Best Epoch", "--"),
        ]
        for key, label, default in card_defs:
            card = tk.Frame(cards_frame, bg=SURFACE, padx=15, pady=8,
                            highlightbackground=OVERLAY, highlightthickness=1)
            card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=3)
            tk.Label(card, text=label, bg=SURFACE, fg="#a6adc8",
                     font=("Segoe UI", 9)).pack(anchor=tk.W)
            val_label = tk.Label(card, text=default, bg=SURFACE, fg=ACCENT,
                                  font=("Segoe UI", 18, "bold"))
            val_label.pack(anchor=tk.W)
            self.cards[key] = val_label

        # ── Charts area ─────────────────────────────────────────────
        charts_frame = ttk.Frame(main, style="Dark.TFrame")
        charts_frame.pack(fill=tk.BOTH, expand=True)

        # Left: Loss + Accuracy plots
        self.fig = Figure(figsize=(8, 5), facecolor=BG_COLOR)
        self.fig.subplots_adjust(hspace=0.35, left=0.08, right=0.97,
                                  top=0.95, bottom=0.08)

        self.ax_loss = self.fig.add_subplot(211)
        self.ax_acc = self.fig.add_subplot(212)

        for ax in [self.ax_loss, self.ax_acc]:
            ax.set_facecolor(SURFACE)
            ax.tick_params(colors=FG_COLOR, labelsize=8)
            ax.spines["bottom"].set_color(OVERLAY)
            ax.spines["left"].set_color(OVERLAY)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.xaxis.label.set_color(FG_COLOR)
            ax.yaxis.label.set_color(FG_COLOR)
            ax.title.set_color(FG_COLOR)

        self.ax_loss.set_title("Loss", fontsize=10, pad=5)
        self.ax_acc.set_title("Accuracy (%)", fontsize=10, pad=5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=charts_frame)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right: Log panel
        log_frame = tk.Frame(charts_frame, bg=SURFACE, width=350)
        log_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        log_frame.pack_propagate(False)

        tk.Label(log_frame, text="Training Log", bg=SURFACE, fg=FG_COLOR,
                 font=("Segoe UI", 11, "bold")).pack(anchor=tk.W, padx=10, pady=(8, 4))

        self.log_text = tk.Text(log_frame, bg="#181825", fg=FG_COLOR,
                                 font=("Consolas", 9), wrap=tk.WORD,
                                 borderwidth=0, highlightthickness=0,
                                 state=tk.DISABLED, padx=8, pady=8)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))

        # Configure log text tags
        self.log_text.tag_configure("best", foreground=GREEN)
        self.log_text.tag_configure("phase", foreground=YELLOW)
        self.log_text.tag_configure("header", foreground=ACCENT)

    def _poll_progress(self):
        """Check for progress file updates every second."""
        try:
            if self.progress_file.exists():
                mtime = self.progress_file.stat().st_mtime
                if mtime != self.last_modified:
                    self.last_modified = mtime
                    with open(self.progress_file, "r") as f:
                        self.data = json.load(f)
                    self._update_display()
                    self.status_label.config(text="Live", foreground=GREEN)
            else:
                self.status_label.config(
                    text=f"Waiting for: {self.progress_file.name}",
                    foreground=YELLOW
                )
        except (json.JSONDecodeError, IOError):
            self.status_label.config(text="Reading...", foreground=YELLOW)

        self.root.after(1000, self._poll_progress)

    def _update_display(self):
        """Update all UI elements with current training data."""
        d = self.data
        if not d:
            return

        epochs = d.get("epochs", [])
        if not epochs:
            return

        current = epochs[-1]
        total_epochs = d.get("total_epochs", "?")
        current_epoch = current.get("epoch", len(epochs))

        # Update cards
        self.cards["epoch"].config(text=f"{current_epoch} / {total_epochs}")

        phase = current.get("phase", "")
        phase_text = "Head (frozen)" if "head" in phase.lower() else "Fine-tune (full)"
        self.cards["phase"].config(text=phase_text)

        train_acc = current.get("train_acc", 0)
        val_acc = current.get("val_acc", 0)
        self.cards["train_acc"].config(text=f"{train_acc:.1f}%")
        self.cards["val_acc"].config(text=f"{val_acc:.1f}%")

        best_val = d.get("best_val_acc", 0)
        best_ep = d.get("best_epoch", "--")
        self.cards["best_val"].config(text=f"{best_val:.1f}%")
        self.cards["best_epoch"].config(text=str(best_ep))

        # Color the val_acc card green if it's the best
        if abs(val_acc - best_val) < 0.01:
            self.cards["val_acc"].config(fg=GREEN)
        else:
            self.cards["val_acc"].config(fg=ACCENT)

        # Update charts
        ep_nums = [e["epoch"] for e in epochs]
        train_losses = [e.get("train_loss", 0) for e in epochs]
        val_losses = [e.get("val_loss", 0) for e in epochs]
        train_accs = [e.get("train_acc", 0) for e in epochs]
        val_accs = [e.get("val_acc", 0) for e in epochs]

        # Loss plot
        self.ax_loss.clear()
        self.ax_loss.set_facecolor(SURFACE)
        self.ax_loss.plot(ep_nums, train_losses, color=ACCENT, linewidth=1.5,
                          label="Train", marker="o", markersize=3)
        self.ax_loss.plot(ep_nums, val_losses, color=RED, linewidth=1.5,
                          label="Val", marker="s", markersize=3)
        self.ax_loss.set_title("Loss", fontsize=10, pad=5, color=FG_COLOR)
        self.ax_loss.legend(fontsize=8, facecolor=SURFACE, edgecolor=OVERLAY,
                            labelcolor=FG_COLOR)
        self.ax_loss.tick_params(colors=FG_COLOR, labelsize=8)
        self.ax_loss.spines["bottom"].set_color(OVERLAY)
        self.ax_loss.spines["left"].set_color(OVERLAY)
        self.ax_loss.spines["top"].set_visible(False)
        self.ax_loss.spines["right"].set_visible(False)
        self.ax_loss.set_xlabel("Epoch", fontsize=8, color=FG_COLOR)

        # Mark phase transition
        frozen_epochs = d.get("epochs_frozen", 5)
        if frozen_epochs < len(ep_nums):
            self.ax_loss.axvline(x=frozen_epochs + 0.5, color=YELLOW,
                                  linestyle="--", alpha=0.5, linewidth=1)
            self.ax_acc.axvline(x=frozen_epochs + 0.5, color=YELLOW,
                                 linestyle="--", alpha=0.5, linewidth=1)

        # Accuracy plot
        self.ax_acc.clear()
        self.ax_acc.set_facecolor(SURFACE)
        self.ax_acc.plot(ep_nums, train_accs, color=ACCENT, linewidth=1.5,
                         label="Train", marker="o", markersize=3)
        self.ax_acc.plot(ep_nums, val_accs, color=GREEN, linewidth=1.5,
                         label="Val", marker="s", markersize=3)

        # Mark best epoch
        if best_ep and best_ep != "--":
            self.ax_acc.axvline(x=best_ep, color=GREEN, linestyle=":",
                                alpha=0.6, linewidth=1)
            self.ax_acc.annotate(f"Best: {best_val:.1f}%",
                                 xy=(best_ep, best_val),
                                 fontsize=7, color=GREEN,
                                 ha="center", va="bottom",
                                 xytext=(0, 8),
                                 textcoords="offset points")

        self.ax_acc.set_title("Accuracy (%)", fontsize=10, pad=5, color=FG_COLOR)
        self.ax_acc.legend(fontsize=8, facecolor=SURFACE, edgecolor=OVERLAY,
                           labelcolor=FG_COLOR)
        self.ax_acc.tick_params(colors=FG_COLOR, labelsize=8)
        self.ax_acc.spines["bottom"].set_color(OVERLAY)
        self.ax_acc.spines["left"].set_color(OVERLAY)
        self.ax_acc.spines["top"].set_visible(False)
        self.ax_acc.spines["right"].set_visible(False)
        self.ax_acc.set_xlabel("Epoch", fontsize=8, color=FG_COLOR)

        # Phase transition line for accuracy too
        if frozen_epochs < len(ep_nums):
            self.ax_acc.axvline(x=frozen_epochs + 0.5, color=YELLOW,
                                 linestyle="--", alpha=0.5, linewidth=1)

        self.canvas.draw_idle()

        # Update log
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)

        # Write header
        model = d.get("model", "EfficientNet-B0")
        dataset_size = d.get("dataset_size", "?")
        self.log_text.insert(tk.END, f"Model: {model}\n", "header")
        self.log_text.insert(tk.END, f"Dataset: {dataset_size} images\n", "header")
        self.log_text.insert(tk.END, f"{'='*40}\n\n", "header")

        for e in epochs:
            ep = e["epoch"]
            phase = e.get("phase", "")

            # Phase header
            if ep == 1:
                self.log_text.insert(tk.END,
                    f"Phase 1: Head Training\n", "phase")
            elif ep == frozen_epochs + 1:
                self.log_text.insert(tk.END,
                    f"\nPhase 2: Fine-tuning\n", "phase")

            line = (f"Ep {ep:>2}: "
                    f"loss={e.get('train_loss', 0):.4f} "
                    f"acc={e.get('train_acc', 0):.1f}% | "
                    f"val_loss={e.get('val_loss', 0):.4f} "
                    f"val_acc={e.get('val_acc', 0):.1f}%")

            is_best = (ep == best_ep)
            tag = "best" if is_best else ""
            suffix = "  << BEST" if is_best else ""
            self.log_text.insert(tk.END, line + suffix + "\n", tag)

        # Show status
        status = d.get("status", "training")
        if status == "completed":
            test_acc = d.get("test_accuracy", "?")
            self.log_text.insert(tk.END,
                f"\nTraining Complete!\n", "phase")
            self.log_text.insert(tk.END,
                f"Test Accuracy: {test_acc}%\n", "best")
            self.status_label.config(text="Training Complete", foreground=GREEN)
        elif status == "early_stopped":
            self.log_text.insert(tk.END,
                f"\nEarly stopping triggered.\n", "phase")

        self.log_text.config(state=tk.DISABLED)
        self.log_text.see(tk.END)


def main():
    parser = argparse.ArgumentParser(description="Training Monitor GUI")
    parser.add_argument("--progress-file", type=str,
                        default=str(DEFAULT_PROGRESS),
                        help="Path to training_progress.json")
    args = parser.parse_args()

    root = tk.Tk()
    root.iconname("OraScan Training Monitor")
    TrainingMonitor(root, args.progress_file)
    root.mainloop()


if __name__ == "__main__":
    main()
