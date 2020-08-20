import prody


def safe_read_ag(ag):
    if isinstance(ag, prody.AtomGroup):
        return ag
    return prody.parsePDB(ag)

