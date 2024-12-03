from setuptools import setup, find_packages

setup(
    name="Jueru",
    version="0.1",
    packages=[package for package in find_packages() if package.startswith("jueru")],
    author='Zihang Wang, Jiayuan Li, Dunqi Yao',
    url='https://github.com/OuAzusaKou/Rl_lib',
    license='MIT',
    include_package_data=True,
    package_data={
        '': ['*.py'],
    }
)
