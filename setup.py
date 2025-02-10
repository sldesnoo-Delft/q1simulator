from setuptools import setup, find_packages

packages = ['q1simulator']
print('packages: {}'.format(packages))

setup(name="q1simulator",
	version="0.15.1",
    description="Simulator for Q1ASM",
    author="Sander de Snoo",
	packages = find_packages(),
    python_requires=">=3.10",
	install_requires=[
        'qcodes',
        'qblox_instruments',
        'numpy',
      ],
    package_data={
        "q1simulator": ["*.hdf5"],
        },
    license='MIT',
	)
