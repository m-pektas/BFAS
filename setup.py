from setuptools import setup, find_packages


with open('bfas/env/requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='bfas',
    version='0.0.4',
    author='Muhammed Pektas',
    description='Brute Force Architecture Search',
    long_description='Brute Force Architecture Search for Neural Architectures to obtain models that have best performences on your device.',
    url='https://github.com/m-pektas/BFAS',
    keywords='bfas, efficient deep learning, NAS, neural architecture search, knowledge distillation',
    python_requires='>=3.7, <4',
    install_requires=required,
    packages=find_packages(),
    package_data={p: ["*"] for p in find_packages()},
    include_package_data=True,
)