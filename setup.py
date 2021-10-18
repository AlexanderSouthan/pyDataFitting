from setuptools import setup, find_packages

setup(
    name='pyRegression',
    version='0.0.1',
    packages=find_packages(where='src'),
    install_requires=['numpy', 'pandas', 'scipy', 'numbers', 'matplotlib', 'scikit-learn', 'statsmodels', 'little_helpers'],
    # dependency_links=['http://github.com/user/repo/tarball/master#egg=package-1.0']
)
