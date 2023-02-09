# This file is part of PIQP.
#
# Copyright (c) 2023 EPFL
# Copyright (c) 2022 INRIA
#
# This source code is licensed under the BSD 2-Clause License found in the
# LICENSE file in the root directory of this source tree.

from . import instruction_set


def load_main_module(globals):
    def load_module(main_module_name):
        import importlib

        try:
            main_module = importlib.import_module("." + main_module_name, __name__)
            globals.update(main_module.__dict__)
            del globals[main_module_name]
            return True
        except ModuleNotFoundError:
            return False

    all_modules = [
        ("piqp_python_avx512", instruction_set.avx512f),
        ("piqp_python_avx2", instruction_set.avx2),
    ]

    for module_name, check in all_modules:
        if check and load_module(module_name):
            return

    assert load_module("piqp_python")


load_main_module(globals=globals())
del load_main_module
