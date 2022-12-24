from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database

def dbEngine(protocol, user, password, host, db_name):
	"""
	Function to create engine for database
	-----------------
	protocol: database manager 
	user: name of the user
	password: passwrord of user
	host: host of db
	db_name: name of database

	"""
	try:
		engine = create_engine(f"{protocol}://{user}:{password}@{host}/{db_name}")
		if not database_exists(engine.url):
			create_database(engine.url)
		return engine
	except NoSuchModuleError as moderr:
		print('Database management system is incorrect.')
	except OperationalError as opererr:
		print('SQL initials are not correct')


class NotFittedError(Exception):
	"""
	User specified exception to handel cases where data was not fitted
	"""
	def __init__(self):
		super().__init__("Data has not been fitted yet!")

	def __str__(self):
		return 'Please, fit the data!'