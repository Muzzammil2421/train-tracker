import setuptools


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
setuptools.setup(
    name='train-tracker',
    version='0.0.1',
    author='Ahmed badr',
    author_email='ahmed.k.badr.97@gmail.com',
    description='library you can use to track your training data while you are training your pytorch model',
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    url='https://github.com/ahmedbadr97/train-tracker',
    license='MIT',
    packages=['traintracker'],
    package_dir={
        'traintracker': 'src/traintracker'},
    install_requires=['torch','numpy','pandas'],
)