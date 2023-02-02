from setuptools import setup

setup(
    name="sourcetracker3",
    author="Gibraan Rahman",
    packages=["st3"],
    include_package_data=True,
    package_data={"": ["*.stan"]}
)
