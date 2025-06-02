from setuptools import setup, find_packages

setup(name='mbd',
    author="Taekyung Kim",
    author_email="taekyung.kim@toyota.com",
    packages=find_packages(include="mdb"),
    version='0.0.1',
    install_requires=[
        'gym', 
        'pandas', 
        'seaborn', 
        'matplotlib', 
        'imageio',
        'control', 
        'tqdm', 
        'tyro', 
        'meshcat', 
        'sympy', 
        'gymnax',
        'jax', 
        'distrax', 
        'gputil', 
        'jaxopt'
        ]
)