import sys
import time
import subprocess
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- Configuration ---
CSV_FILE_TO_WATCH = "experiment_results.csv"
PLOTTER_SCRIPT = "plot_results.py"
# -------------------

# Global variable to hold the plot process so we can manage it
plot_process = None

class PlotUpdateHandler(FileSystemEventHandler):
    """A handler for file system events that relaunches the plot when the CSV is modified."""
    def on_modified(self, event):
        # This event is triggered on file modification.
        # We check if the modified file is the one we're watching.
        if not event.is_directory and os.path.basename(event.src_path) == CSV_FILE_TO_WATCH:
            print(f"Detected update in '{CSV_FILE_TO_WATCH}'. Relaunching plot...")
            relaunch_plot()

def relaunch_plot():
    """Terminates the existing plot process and launches a new one."""
    global plot_process
    # Terminate the existing plot process if it's running
    if plot_process:
        plot_process.terminate()
        plot_process.wait()  # Wait for the process to actually close to avoid zombie processes

    # Launch a new plot process using the same Python interpreter
    plot_process = subprocess.Popen([sys.executable, PLOTTER_SCRIPT])

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