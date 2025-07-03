from setuptools import setup
from pathlib import Path

version = Path(__file__).with_name("VERSION.txt").read_text().strip()

setup(
    name='gati',
    version=version,
    py_modules=['gati'],
    package_dir={'': 'python'},
    install_requires=[
        'numpy',
    ],
)

