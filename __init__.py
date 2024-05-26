import os

# Set PYFLEXROOT to the current working directory plus /PyFlex
pyflex_root = os.path.join('/home/jiale/softgym/PyFlex')
os.environ['PYFLEXROOT'] = pyflex_root

# Update PYTHONPATH to include the PyFlex bindings build directory
bindings_build_path = os.path.join(pyflex_root, 'bindings', 'build')
os.environ['PYTHONPATH'] = f"{bindings_build_path}:{os.environ.get('PYTHONPATH', '')}"

# Update LD_LIBRARY_PATH to include the SDL2 library directory
sdl2_lib_path = os.path.join(pyflex_root, 'external', 'SDL2-2.0.4', 'lib', 'x64')
os.environ['LD_LIBRARY_PATH'] = f"{sdl2_lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"
