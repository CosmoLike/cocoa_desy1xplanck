"""Loader stub for the compiled cosmolike library of desy1xplanck.

The real module is the compiled C++ extension
cosmolike_desy1xplanck_interface.so in this folder, which
scripts/compile_desy1xplanck.sh builds from interface.cpp (the pybind11
bindings of the cosmolike C code). In each folder, Python's import looks
for an extension module (.so) before a .py file of the same name, so with
the .so next to this file, `import cosmolike_desy1xplanck_interface`
loads the compiled library directly and this file does not run. It runs
only when Python finds it first; it then loads the .so by hand and puts
the compiled module in place of this one.

This is the stub layout that setuptools writes next to a compiled
extension. It relies on pkg_resources and on the imp module, which
Python 3.12 removed.
"""

def __bootstrap__():
   """Load the .so of this folder in place of this stub module.

   pkg_resources.resource_filename returns the path of the .so next to
   this module; imp.load_dynamic loads it under this module's name, which
   replaces the entry of sys.modules, so the importer receives the
   compiled module. The global statement lets the function rebind the
   module-level names __file__ and __loader__; del then removes the
   helper names __bootstrap__ and __loader__ from the module.
   """
   global __bootstrap__, __loader__, __file__
   import sys, pkg_resources, imp
   __file__ = pkg_resources.resource_filename(__name__,'cosmolike_desy1xplanck_interface.so')
   __loader__ = None; del __bootstrap__, __loader__
   imp.load_dynamic(__name__,__file__)
# runs the loader once, when Python executes this file
__bootstrap__()
