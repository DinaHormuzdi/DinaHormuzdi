"""Launch Dina Bot and open in the browser."""

import threading
import webbrowser
import time
from backend.dina_bot import run_server, DEFAULT_PORT


def _start_backend():
    run_server(port=DEFAULT_PORT)


if __name__ == "__main__":
    t = threading.Thread(target=_start_backend, daemon=True)
    t.start()
    time.sleep(1)
    webbrowser.open(f"http://localhost:{DEFAULT_PORT}")
    t.join()
