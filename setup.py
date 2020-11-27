from setuptools import setup, find_packages


setup(
    name="keithley2600",
    version="v2.0.1",
    description="Full Python driver for the Keithley 2600 series.",
    url="https://github.com/OE-FET/keithley2600.git",
    author="Sam Schott",
    author_email="ss2151@cam.ac.uk",
    licence="MIT",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "pyvisa",
        "pyvisa-py",
        "numpy",
    ],
    zip_safe=False,
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
