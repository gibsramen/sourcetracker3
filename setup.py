from setuptools import setup, find_packages

setup(
    name="sourcetracker3",
    author="Gibraan Rahman",
    packages=find_packages(),
    include_package_data=True,
    package_data={"": ["*.stan"]},
    data_files=[("st3/tests/test_data", ["table.biom", "metadata.tsv"])]
)
