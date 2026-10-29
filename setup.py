from setuptools import setup, find_packages

setup(
    name='sag',
    version='1.0.0',
    description='SAG: Stable Anomaly Detection in Dynamic Graphs',
    author='Anonymous',
    python_requires='>=3.8',
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'scikit-learn>=1.0.0',
        'pandas>=1.3.0',
        'matplotlib>=3.5.0',
        'seaborn>=0.11.0',
        'networkx>=2.6.0',
        'tensorboard>=2.8.0',
        'tqdm>=4.62.0',
        'psutil>=5.8.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'pytest-cov>=2.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'sag-train=scripts.train_single:main',
            'sag-inference=scripts.inference:main',
            'sag-ablation=scripts.ablation_study:main',
        ],
    },
)
