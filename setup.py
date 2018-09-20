from setuptools import setup, find_packages

setup(name="Keithley2600",
      version="0.1.0",
      description="Full Python driver for the Keithley 2600 series.",
      url="https://github.com/OE-FET/Keithley2600-driver.git",
      author="Sam Schott",
      author_email="ss2151@cam.ac.uk",
      licence='MIT',
      long_description=open('README.md').read(),
      packages=find_packages(),
      install_requires=[
          'PyVISA',
          'pyvisa-py',
          'QtPy',
          'setuptools',
          'numpy',
          'matplotlib',
      ],
      zip_safe=False,
)
