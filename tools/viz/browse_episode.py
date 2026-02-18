"""
Browse frames of a LeRobot episode in a Tkinter window.

Usage:
    python tools/viz/browse_episode.py --repo_id=local/lerobot_cup_and_saucer --episode=0

Controls:
    Right Arrow / Space  → Next frame
    Left Arrow           → Previous frame
    Home                 → Jump to first frame
    End                  → Jump to last frame
    Escape               → Quit
"""

import argparse
import tkinter as tk

import numpy as np
from PIL import Image, ImageTk

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402


def load_episode_indices(dataset, episode_idx):
    """Return (from_idx, to_idx) for the given episode."""
    if dataset.meta.episodes is None:
        try:
            from lerobot.datasets.utils import load_episodes

            dataset.meta.episodes = load_episodes(dataset.root)
        except Exception as e:
            raise RuntimeError(f"Could not load episodes metadata: {e}") from e

    total = len(dataset.meta.episodes)
    if episode_idx < 0 or episode_idx >= total:
        raise ValueError(f"Episode {episode_idx} out of range (0-{total - 1})")

    ep = dataset.meta.episodes[episode_idx]
    from_idx = int(
        ep["dataset_from_index"]
        if not isinstance(ep["dataset_from_index"], list)
        else ep["dataset_from_index"][0]
    )
    to_idx = int(
        ep["dataset_to_index"] if not isinstance(ep["dataset_to_index"], list) else ep["dataset_to_index"][0]
    )
    return from_idx, to_idx


def numpy_from_item(val):
    """Convert a dataset value to a numpy array."""
    if hasattr(val, "numpy"):
        return val.numpy()
    return np.array(val)


def format_vector(arr, label, max_per_line=7):
    """Format a numpy vector as a readable multi-line string."""
    vals = arr.flatten().tolist()
    lines = [f"{label}  (len={len(vals)})"]
    for start in range(0, len(vals), max_per_line):
        chunk = vals[start : start + max_per_line]
        lines.append("  " + "  ".join(f"{v:+8.4f}" for v in chunk))
    return "\n".join(lines)


class EpisodeBrowser:
    def __init__(self, dataset, episode_idx, from_idx, to_idx):
        self.dataset = dataset
        self.episode_idx = episode_idx
        self.from_idx = from_idx
        self.to_idx = to_idx
        self.num_frames = to_idx - from_idx
        self.current = 0  # relative index within the episode

        # --- Tkinter setup ---
        self.root = tk.Tk()
        self.root.title(f"Episode {episode_idx}  —  {self.num_frames} frames")
        self.root.geometry("1200x850")
        self.root.configure(bg="#1e1e1e")

        # ---- Top: frame counter ----
        self.header = tk.Label(
            self.root,
            text="",
            font=("Consolas", 13, "bold"),
            bg="#1e1e1e",
            fg="#e0e0e0",
            anchor="center",
        )
        self.header.pack(side="top", fill="x", pady=(8, 2))

        # ---- Middle: camera images ----
        self.cam_container = tk.Frame(self.root, bg="#1e1e1e")
        self.cam_container.pack(side="top", fill="both", expand=True, padx=8, pady=4)
        self.cam_container.pack_propagate(False)

        # We'll create camera sub-frames dynamically on first load
        self._cam_labels: dict[str, tk.Label] = {}
        self._cam_title_labels: dict[str, tk.Label] = {}
        self._cam_frames: dict[str, tk.Frame] = {}
        self._cam_keys: list[str] | None = None  # filled on first frame

        # ---- Bottom: text info (observation + action) ----
        self.text_frame = tk.Frame(self.root, bg="#1e1e1e")
        self.text_frame.pack(side="bottom", fill="x", padx=8, pady=(0, 8))

        self.info_text = tk.Text(
            self.text_frame,
            height=12,
            font=("Consolas", 10),
            bg="#2d2d2d",
            fg="#c8c8c8",
            relief="flat",
            wrap="none",
            state="disabled",
        )
        self.info_text.pack(fill="both", expand=True)

        # ---- Key bindings ----
        self.root.bind("<Right>", lambda e: self.step(1))
        self.root.bind("<space>", lambda e: self.step(1))
        self.root.bind("<Left>", lambda e: self.step(-1))
        self.root.bind("<Home>", lambda e: self.jump(0))
        self.root.bind("<End>", lambda e: self.jump(self.num_frames - 1))
        self.root.bind("<Escape>", lambda e: self.root.destroy())

        # Initial render
        self.show_frame()

    # ------------------------------------------------------------------
    def step(self, delta):
        new = self.current + delta
        if 0 <= new < self.num_frames:
            self.current = new
            self.show_frame()

    def jump(self, idx):
        self.current = max(0, min(idx, self.num_frames - 1))
        self.show_frame()

    # ------------------------------------------------------------------
    def show_frame(self):
        idx = self.from_idx + self.current
        self.header.config(text=f"Frame {self.current} / {self.num_frames - 1}   (dataset idx {idx})")

        try:
            item = self.dataset[idx]
        except Exception as e:
            self._set_info(f"Error loading frame {idx}: {e}")
            return

        # --- Discover camera keys on first call ---
        if self._cam_keys is None:
            self._cam_keys = sorted(k for k in item if "image" in k)
            for _, key in enumerate(self._cam_keys):
                clean = key.replace("observation.images.", "")
                frame = tk.Frame(self.cam_container, bd=1, relief="solid", bg="#2d2d2d")
                frame.pack(side="left", expand=True, fill="both", padx=2, pady=2)
                frame.pack_propagate(False)

                title = tk.Label(frame, text=clean, font=("Arial", 10, "bold"), bg="#2d2d2d", fg="#aaa")
                title.pack(side="top")

                lbl = tk.Label(frame, bg="#2d2d2d")
                lbl.pack(expand=True, fill="both")

                self._cam_frames[key] = frame
                self._cam_title_labels[key] = title
                self._cam_labels[key] = lbl

        # --- Update camera images ---
        for key in self._cam_keys:
            if key not in item:
                continue
            img_data = item[key]
            arr = numpy_from_item(img_data)
            if arr.ndim == 3 and arr.shape[0] <= 4:
                arr = np.transpose(arr, (1, 2, 0))
            if arr.dtype == np.float32 or arr.dtype == np.float64:
                arr = (arr * 255).clip(0, 255).astype(np.uint8)

            pim = Image.fromarray(arr)

            lbl = self._cam_labels[key]
            lbl.update_idletasks()
            cw = max(lbl.winfo_width(), 100)
            ch = max(lbl.winfo_height(), 100)
            ratio = min(cw / pim.width, ch / pim.height)
            new_w, new_h = int(pim.width * ratio), int(pim.height * ratio)
            if new_w > 0 and new_h > 0:
                pim = pim.resize((new_w, new_h), Image.Resampling.LANCZOS)

            photo = ImageTk.PhotoImage(pim)
            lbl.config(image=photo)
            lbl.image = photo  # prevent GC

        # --- Build text for observation & action ---
        lines = []

        # Observation state
        for key in sorted(item.keys()):
            if "observation.state" in key or "observation.pos" in key:
                arr = numpy_from_item(item[key])
                lines.append(format_vector(arr, key))

        # Action
        if "action" in item:
            arr = numpy_from_item(item["action"])
            lines.append(format_vector(arr, "action"))

        self._set_info("\n".join(lines) if lines else "(no vector data)")

    # ------------------------------------------------------------------
    def _set_info(self, text):
        self.info_text.config(state="normal")
        self.info_text.delete("1.0", "end")
        self.info_text.insert("1.0", text)
        self.info_text.config(state="disabled")

    def run(self):
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Browse LeRobot episode frames in a Tkinter window.")
    parser.add_argument(
        "--repo_id", type=str, required=True, help="Dataset repo id, e.g. local/lerobot_cup_and_saucer"
    )
    parser.add_argument("--episode", type=int, required=True, help="Episode index")
    parser.add_argument(
        "--root",
        type=str,
        default="/home/droplab/.cache/huggingface/lerobot/",
        help="Dataset root directory override",
    )
    args = parser.parse_args()

    print(f"Loading dataset: {args.repo_id}")

    repo_id = args.repo_id

    print(repo_id, args.root + repo_id)

    dataset = LeRobotDataset(repo_id, root=args.root + repo_id)

    from_idx, to_idx = load_episode_indices(dataset, args.episode)
    print(f"Episode {args.episode}: frames {from_idx}..{to_idx}  ({to_idx - from_idx} total)")

    browser = EpisodeBrowser(dataset, args.episode, from_idx, to_idx)
    browser.run()


if __name__ == "__main__":
    main()
