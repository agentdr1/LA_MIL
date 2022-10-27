from setuptools import setup, find_packages

setup(
  name = 'tmil',
  packages = find_packages(),
  version = '0.0.1',
  license='MIT',
  description = 'Transformer based Multiple Instance Learning',
  author = 'Daniel Reisenbüchler',
  author_email = 'hpc.agent.dr@gmail.com',
  url = 'https://github.com/agentdr1/LA_MIL',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'multiple instance learning',
    'whole slide image',
    'genetic alteration prediction',      
    'local-attention'
  ],
  install_requires=[
    'einops>=0.3',
    'torch>=1.6',
    'dgl>=0.9.1',
    'scikit-learn>=1.1.1',
    'numpy>=1.23.1'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.9',
  ],
)