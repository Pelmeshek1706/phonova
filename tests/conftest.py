import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OPENWILLIS_SPEECH_SRC = REPO_ROOT / "openwillis" / "openwillis-speech" / "src"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if str(OPENWILLIS_SPEECH_SRC) not in sys.path:
    sys.path.insert(0, str(OPENWILLIS_SPEECH_SRC))

import pytest

from tests.helpers.data_loading import normalize_language


def pytest_addoption(parser):
    parser.addoption(
        "--language",
        action="store",
        default=None,
        type=normalize_language,
        help="Limit staged language-specific tests to en or ua (aliases: en/eng, ua/uk/ukr).",
    )


@pytest.fixture(scope="session")
def language(request):
    return request.config.getoption("--language")
