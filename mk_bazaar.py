""" mk_bazaar.py

    Output instance for the 2D knapsack problem with splittable items and stability issues
    in the format used by
    F.Furini,E.Malaguti,D.Thomopulos.
    Modeling Two-Dimensional Guillotine Cutting Problems via Integer Programming. 
    INFORMS Journal on Computing, 28(4) (2016) 603-799.

    Usage:
    python3 mk_bazaar.py seed n
      where:
        - 'seed' initilizes the random generator
        - 'n' is the number of items
"""

import random
import math
import sys


def mk_instance(n, H, W, h, w, u):
    """Print instance data in the standard output

    Parameters:
      n - number of classes (in our case, equalt to the number of items)
      H - bin height
      W - bin width
      h - h[i] - height of class/item i
      w - w[i] - width of class/item i
      u - u[i] - unit utility of class/item i
    """
    ITEMS = range(n)
    print(f"itemNumber {n}")
    print(f"itemClassNumber {n}")
    print(f"itemClass_widthVector", " ".join([f"{w[i]}" for i in ITEMS]))
    print(f"itemClass_lengthVector", " ".join([f"{h[i]}" for i in ITEMS]))
    print(f"itemClass_demand", " ".join(["1" for i in ITEMS]))
    print(f"binNumber 10000")  # not used, setting the same value as Furini
    print(f"binClassNumber 1")
    print(f"binClass_widthVector {W}")
    print(f"binClass_lengthVector {H}")
    print(f"binClass_disponibility 1000")  # not used, setting the same value as Furini
    print(f"itemClass_weight", " ".join([f"{u[i]*h[i]}" for i in ITEMS]))


def mk_bazaar(n=100):
    """create instance for the 2D knapsack problem with splittable items and stability issues

       Returns:
       tuple (n, H, W, h, w, u), as described in "mk_instance"
    """
    W = 100 * n // 100   # make box with width 100 for n=100, otherwise proportional to n
    H = W                # square box
    U = 1000             # maximum utility
    height = [0 for _ in range(n)]
    width = [0 for _ in range(n)]
    utility = [0 for _ in range(n)]

    # fill container with n//2 items
    # first, create basis with ~sqrt(n//2) stacks
    B = W/math.sqrt(n//2)
    bases = [random.randint(int(1/1.1*B), int(1.1*B)) for _ in range(1+int(math.sqrt(n//2)))]
    S = sum(bases)
    s = 0
    bases.pop()
    for i in range(int(math.sqrt(n//2))):
        s += bases[i]
        bases[i] = max(1,int(.5+W*s/S))
    bases = sorted(bases)
    height, width, utility = [], [], []
    stacks = []
    for i in range(len(bases)+1):
        if i == 0:
            b = bases[i]
        elif i == len(bases):
            b = H-bases[-1]
        else:
            b = bases[i] - bases[i-1]
        stack = set()
        while len(stack) < int(math.sqrt(n//2)):
            stack.add(random.randint(1,H-1))
        stack = sorted(stack)

        w, h, u = [], [], []
        ui = int(U * b / W   * n // 100)
        for j in range(len(stack)+1):
            if j == 0:
                h.append(stack[j])
                w.append(b)
            else:
                if j == len(stack):
                    h.append(H-stack[-1])
                else:
                    h.append(stack[j] - stack[j-1])
                w.append(w[-1] - 1)
            random.randint(ui,ui) # unused ...
            u.append(ui)
        height.extend(h)
        width.extend(w)
        utility.extend(u)

    # fill set of items with small variations on the previously created ones
    while len(width) < n:
        k = random.choice(range(int(math.sqrt(n//2))))    # choice among items of the first step
        h = random.randint(height[k]-1, height[k]+1)
        w = width[k] + 1 + random.randint(0,int(0.1*width[k]))
        u = utility[k] + 1 + random.randint(0,int(0.1*utility[k]))
        height.append(h if h > 0 else 1)
        width.append(w if w > 0 else 1)
        utility.append(u if u > 0 else 1)

    return n, H, W, height, width, utility



if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} seed n\n"
               "  where:\n"
               "    - 'seed' initilizes the random generator\n"
               "    - 'n' is the number of items")
        exit(-1)
    seed = int(sys.argv[1])
    random.seed(seed)
    n = int(sys.argv[2])
    n, H, W, height, width, utility = mk_bazaar(n)
    mk_instance(n, H, W, height, width, utility)
