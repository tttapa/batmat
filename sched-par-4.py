lP = 6

RED = "\x1b[1;31m"
GREEN = "\x1b[1;32m"
YELLOW = "\x1b[1;33m"
BLUE = "\x1b[1;34m"
PINK = "\x1b[1;35m"
RST = "\x1b[0m"


def countr_zero(n: int) -> int:
    if n == 0:
        raise ValueError("countr_zero is undefined for zero")
    return (n & -n).bit_length() - 1


def is_active(l, bi):
    lbi = lP if bi == 0 else countr_zero(bi)
    inactive = lbi < l
    return ((bi >> l) & 1) == 1 and not inactive


def run(l, ti):
    offset = 1 << l
    bi = (ti - offset) % (1 << lP)
    biU = ti # (bi + offset) % (1 << lP)
    active = is_active(l, bi)
    do_U = is_active(l, biU)
    U_below_Y = ((bi >> l) & 3) == 1 and l + 1 != lP

    if active:
        job = f"{YELLOW}{bi:>2}{RST}"
        clr = GREEN if U_below_Y else YELLOW
        biUY = ((bi + offset) if U_below_Y else (bi - offset)) % (1 << lP)
        job += f"  {BLUE}{bi:>2} --{RST} {clr}{biUY:>2}{RST}   "
    elif do_U:
        job = f"{GREEN}{biU:>2}{RST}"
        biD = (biU - offset) % (1 << lP)
        biY = (biD - offset) % (1 << lP)
        job += f"  {RED}{biU:>2}{RST} {RED}{biY:>2}{RST} {RED}{biD:>2}{RST}"
        if is_active(l + 1, biD):
            job += f" {PINK}{biD:>2}{RST}"
        else:
            job += "   "
    else:
        job = "     "
        job += "          "
    return job


jobs = [[] for _ in range(1 << lP)]
for l in range(lP):
    for ti in range(1 << lP):
        jobs[ti] += [run(l, ti)]
for j in jobs:
    print("  |  ".join(j))
