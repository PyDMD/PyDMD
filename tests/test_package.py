from unittest import TestCase
import unittest
import pkgutil
from os import walk
from os import path


class TestPackage(TestCase):
	def test_modules_name(self):
		# it checks that __all__ includes all the .py files in dmd folder
		import dmd
		package = dmd

		f_aux = []
		for (__, __, filenames) in walk('dmd'):
			f_aux.extend(filenames)

		f = []
		for i in f_aux:
			file_name, file_ext = path.splitext(i)
			if file_name != '__init__' and file_ext == '.py':
				f.append(file_name)

		assert (sorted(package.__all__) == sorted(f))

	def test_import_dm_1(self):
		import dmd as dm
		dmd = dm.dmd.DMDBase()

	def test_import_dm_2(self):
		import dmd as dm
		dmd = dm.dmd.DMD()

	def test_import_dm_3(self):
		import dmd as dm
		dmd = dm.fbdmd.FbDMD()
