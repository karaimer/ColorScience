#!/usr/bin/env python
#
# Author: Ying Xiong.
# Created: Jun 20, 2013.

"""Some extended utility functions for 'os' module."""

import errno
import glob
import shutil
import os

def mkdir_p(path, mode = 0o777):
    """Create a leaf directory 'path' and all intermediate ones.

    No error will be reported if the directory already exists.  Same effect as
    the unix command 'mkdir -p path'.

    """
    if (not os.path.isdir(path)):
        os.makedirs(path, mode)

def rm_rf(path):
    """Remove a file or a directory, recursively.

    No error will be reported if 'path' does not exist. The 'path' can be a list
    or tuple.  Same effect as the unix command 'rm -rf path'.

    """
    if (type(path) in [list, tuple]):
        for p in path:
            rm_rf(p)
    else:
        if (os.path.exists(path)):
            if (os.path.isdir(path)):
                shutil.rmtree(path)
            else:
                os.remove(path)

def cp_r(src, dst):
    """
    Same effect as the unix command 'cp -r src dst', supporting the followings:

    #. `cp_r("/path/to/src_file", "/path/to/"dst_file")`:
       The 'src_file' is a single file, and 'dst_file' is created or overwritten
       if already exists.

    #. `cp_r("/path/to/src_folder", "/path/to/dst_folder")`:
       The 'dst_folder' is a single folder, and 'dst_folder' will be created if
       not already exists, otherwise a "/path/to/dst_folder/src_folder" will be
       created.

    #. `cp_r("/path/to/src", "/path/to/dst_folder")`:
       The 'src' can be either a file or a folder, and can contain wildcard
       characters (e.g. '*'), and the 'dst_folder' must already exist.

    #. `cp_r(["/path/to/src1", "/path/to/src2", ...], "/path/to/dst_folder")`:
       The 'src' can be anything as the previous syntax, and the first argument
       can be either list or tuple. The 'dst_folder' must already exist.
    """
    if (not os.path.exists(dst)):
        # Case 1 or 2.
        _do_cp_r(src, dst)
    elif (not os.path.isdir(dst)):
        # Case 1.
        shutil.copy(src, dst)
    else:
        # Case 2, 3, 4.
        if (type(src) in [list, tuple]):
            for s in src:
                cp_r(s, dst)
        else:
            no_match = True
            for f in glob.iglob(src):
                no_match = False
                _do_cp_r(f, os.path.join(dst, os.path.basename(f)))
            if (no_match):
                raise Exception("File '%s' cannot be found.\n" % src)

def _do_cp_r(src, dst):
    """A helper function for cp_r.

    The 'src' has to be a single file or folder, and 'dst' must not already
    exist.

    """
    try:
        shutil.copytree(src, dst)
    except OSError as exc:
        if (exc.errno == errno.ENOTDIR):
            shutil.copy(src, dst)
        else:
            raise
