from setuptools import setup, find_packages


def get_metadata(relpath, varname):
    """Read metadata info from a file without importing it."""
    from os.path import dirname, join

    if "__file__" not in globals():
        root = ".."
    else:
        root = dirname(__file__)

    for line in open(join(root, relpath), "rb"):
        line = line.decode("cp437")
        if varname in line:
            if '"' in line:
                return line.split('"')[1]
            elif "'" in line:
                return line.split("'")[1]


setup(name="keithley2600",
      version=get_metadata("keithley2600/keithley_driver.py", "__version__"),
      description="Full Python driver for the Keithley 2600 series.",
      url="https://github.com/OE-FET/keithley2600.git",
      author="Sam Schott",
      author_email="ss2151@cam.ac.uk",
      licence='MIT',
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      packages=find_packages(),
      install_requires=[
          'PyVISA',
          'pyvisa-py',
          'numpy',
      ],
      zip_safe=False,
)
