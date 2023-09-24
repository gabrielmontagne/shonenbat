from setuptools import setup

setup(
    name='shonenbat',
    author='Gabriel Montagné Láscaris-Comneno',
    author_email='gabriel@tibas.london',
    license='MIT',
    version='0.2.0',
    entry_points={
        'console_scripts': [
            'shonenbat = shonenbat.__main__:main',
            'shonenlist = shonenbat.__main__:list',
            'shonenimage = shonenbat.__main__:image',
            'shonenchat = shonenbat.__main__:chat',
            'shonencount = shonenbat.__main__:count'
        ]
    }
)
