from distutils.core import setup
import io

with io.open('README.rst', mode='r', encoding='utf-8') as desc_file:
    long_description = desc_file.read()

setup(
    name='adaptfilt',
    packages=['adaptfilt'],  # this must be the same as the name above
    version='0.1',
    description='Adaptive filtering module for Python',
    long_description=long_description,
    author='Jesper Wramberg & Mathias Tausen',
    author_email='jesper.wramberg@gmail.com & mathias.tausen@gmail.com',
    url='https://github.com/Wramberg/adaptfilt',  # use the URL to the github repo
    keywords=['adaptive filter', 'adaptive filtering', 'signal-processing', 'lms', 'apa', 'nlms', 'rls'],  # arbitrary keywords
    classifiers=[
        'Development Status :: 1 - Planning',
        'Programming Language :: Python',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    license='MIT',
)
