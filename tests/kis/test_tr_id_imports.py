import importlib


def test_tr_id_modules_import_after_kis_migration():
    for module_name in (
        "kis.tr_id.register",
        "kis.tr_id.call",
        "kis.tr_id.deriv_minute",
    ):
        importlib.import_module(module_name)
