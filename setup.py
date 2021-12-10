from setuptools import setup, find_packages

packages = ['q1simulator']
print('packages: {}'.format(packages))

setup(name="q1simulator",
	version="0.1",
    description="Simulator for Q1ASM",
    author="Sander de Snoo",
	packages = find_packages(),
	install_requires=[
        'qcodes',
      ],
	)
