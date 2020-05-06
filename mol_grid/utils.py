import prody


def safe_read_ag(ag) -> prody.Atomic:
    if isinstance(ag, prody.AtomGroup):
        return ag
    elif isinstance(ag, str):
        return prody.parsePDB(ag)
    else:
        raise RuntimeError(f"Can't read atom group, 'ag' has wrong type {type(ag)}")
