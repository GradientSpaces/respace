from setuptools import setup, find_packages

setup(
    name='stan24sgllm',
    version='0.1',
    packages=find_packages(include=['src', 'src.*']),
)