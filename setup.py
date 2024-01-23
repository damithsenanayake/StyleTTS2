from setuptools import setup, find_packages

setup(
    name='StyleTTS2',
    version='0.3.4',
    packages=find_packages(include=['StyleTTS2Synth*']),
    install_requires=[
        # List your dependencies here
    ],
    entry_points={
        'console_scripts': [
            # Add any console scripts if needed
        ],
    },
)