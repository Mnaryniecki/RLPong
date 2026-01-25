import time
import subprocess
import os
import sys
import glob
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- Configuration ---
CSV_FILE_TO_WATCH = " "
CSV_PATTERN = "results_*.csv"
PLOTTER_SCRIPT = "plot_results.py"
# -------------------

# Global variable to hold the plot process so we can manage it
plot_process = None


class PlotUpdateHandler(FileSystemEventHandler):
    """A handler for file system events that relaunches the plot when the CSV is modified."""
    """A handler for file system events that relaunches the plot when a result CSV is modified."""

    def on_modified(self, event):
        # This event is triggered on file modification.
        # We check if the modified file is the one we're watching.
        if not event.is_directory and os.path.basename(event.src_path) == CSV_FILE_TO_WATCH:
            print(f"Detected update in '{CSV_FILE_TO_WATCH}'. Relaunching plot...")
        filename = os.path.basename(event.src_path)
        if not event.is_directory and filename.startswith("results_") and filename.endswith(".csv"):
            print(f"Detected update in '{filename}'. Relaunching plot...")
            relaunch_plot()

    def on_created(self, event):
        # Also handle creation of new result files
        filename = os.path.basename(event.src_path)
        if not event.is_directory and filename.startswith("results_") and filename.endswith(".csv"):
            print(f"Detected creation of '{filename}'. Relaunching plot...")
            relaunch_plot()


def relaunch_plot():
    """Terminates the existing plot process and launches a new one."""
    global plot_process
    # Terminate the existing plot process if it's running
    if plot_process:
        plot_process.terminate()
        plot_process.wait()  # Wait for the process to actually close to avoid zombie processes
        try:
            plot_process.wait(timeout=1)  # Wait for the process to actually close
        except subprocess.TimeoutExpired:
            plot_process.kill()

    # Find all matching CSV files
    csv_files = glob.glob(CSV_PATTERN)
    if not csv_files:
        print("No result CSV files found yet.")
        return

    # Launch a new plot process using the same Python interpreter
    plot_process = subprocess.Popen([sys.executable, PLOTTER_SCRIPT])
    # Pass all found CSV files as arguments to the plotter
    cmd = [sys.executable, PLOTTER_SCRIPT] + csv_files
    print(f"Launching plotter with: {csv_files}")
    plot_process = subprocess.Popen(cmd)


def main():
    """Sets up the file watcher and runs indefinitely."""
    # Initial launch of the plot when the script starts
    print("Starting initial plot...")
    relaunch_plot()

    # Set up the watchdog observer to monitor the current directory
    path = "."
    event_handler = PlotUpdateHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)

    print(f"Watching for changes in '{CSV_FILE_TO_WATCH}'...")
    print(f"Watching for changes in '{CSV_PATTERN}'...")
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping watcher...")
    finally:
        # Clean up on exit
        if plot_process:
            plot_process.terminate()
        observer.stop()
        observer.join()
        print("Watcher stopped.")


if __name__ == "__main__":
    main()