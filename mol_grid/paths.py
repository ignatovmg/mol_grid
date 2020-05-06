from path import Path

SRC_DIR = Path(__file__).abspath().dirname()
ROOT_DIR = SRC_DIR.dirname()
TEST_DIR = SRC_DIR / 'tests'
TEST_DATA = TEST_DIR / 'data'
