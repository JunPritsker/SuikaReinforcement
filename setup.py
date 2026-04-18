from setuptools import setup, find_packages

setup(
    name="suika_reinforcement",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "gymnasium",
        "numpy",
        "pygame",
        "pymunk",
        "pyyaml",
        "stable-baselines3",
        "shimmy"
    ],
)