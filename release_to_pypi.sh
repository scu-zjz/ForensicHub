# Only for Developers!!!!!!!!!
rm -rf build dist *.egg-info
python setup.py sdist bdist_wheel
twine upload --repository-url https://upload.pypi.org/legacy/ dist/*