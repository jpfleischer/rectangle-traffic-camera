import sys
from pathlib import Path

import dotenv

HERE = Path(__file__).resolve().parent
TRACKSVIEWER_ROOT = HERE.parent  # .../tracksviewer/
REPO_ROOT = TRACKSVIEWER_ROOT.parent  # .../rectangle-traffic-camera/

DOTENV_PATH = REPO_ROOT / ".env"

def init_env_and_sys_path() -> None:
    """
    - loads REPO_ROOT/.env
    - adds REPO_ROOT/roadpairer to sys.path so clickhouse_client import works
    """
    dotenv.load_dotenv(str(DOTENV_PATH), override=False)

    gui_dir = REPO_ROOT / "roadpairer"
    if str(gui_dir) not in sys.path:
        sys.path.insert(0, str(gui_dir))
