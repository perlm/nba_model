#!/usr/bin/python

from distutils.core import setup

setup(name='nba_model',
	version='0.1.0',
	description="NBA pregame prediction model", 
	packages=['nba_model'],
	install_requires=['numpy','pandas','sklearn','twitter'],
	entry_points={
		'console_scripts': [
		'nba_model = nba_model.__main__:main'
		]
	}
	)
