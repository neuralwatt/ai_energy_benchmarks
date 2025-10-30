from setuptools import find_packages, setup

# Read the version from the VERSION file
with open("VERSION.txt") as version_file:
    version = version_file.read().strip()

# Read the requirements from the requirements.txt file
with open("requirements.txt") as requirements_file:
    requirements = requirements_file.read().splitlines()

setup(
    name="ai_energy_benchmarks",
    version=version,
    packages=find_packages(),
    install_requires=requirements,
    # ...existing code...
)
