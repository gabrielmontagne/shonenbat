from setuptools import setup

setup(
    name='shonenbat',
    author='Gabriel Montagné Láscaris-Comneno',
    author_email='gabriel@tibas.london',
    license='MIT',
    version='0.1.0',
    entry_points={
        'console_scripts': [
            'shonenbat = shonenbat.__main__:main'
        ]
    }
)
