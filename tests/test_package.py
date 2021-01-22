from unittest import TestCase
import unittest
import pkgutil
from os import walk
from os import path


class TestPackage(TestCase):
    def test_import_dm_1(self):
        import pydmd as dm
        dmd = dm.dmd.DMDBase()

    def test_import_dm_2(self):
        import pydmd as dm
        dmd = dm.dmd.DMD()

    def test_import_dm_3(self):
        import pydmd as dm
        dmd = dm.fbdmd.FbDMD()
