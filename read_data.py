def read_data(filename):  # read weighted DIMITRI's format
    # infile = sys.argv[1]
    with open(filename) as fin:
        lines = fin.readlines()

        line = lines.pop(0)
        tokens = line.split()
        assert tokens[0] == "itemNumber"
        N = int(tokens[1])  # total number of items
     
        line = lines.pop(0)
        tokens = line.split()
        assert tokens[0] == "itemClassNumber"
        n = int(tokens[1])  # number of different items
     
        line = lines.pop(0)
        tokens = line.split()
        assert tokens[0] == "itemClass_widthVector"
        w = [int(x) for x in tokens[1:]]
     
        line = lines.pop(0)
        tokens = line.split()
        assert tokens[0] == "itemClass_lengthVector"
        h = [int(x) for x in tokens[1:]]
     
        line = lines.pop(0)
        tokens = line.split()
        assert tokens[0] == "itemClass_demand"
        d = [int(x) for x in tokens[1:]]
     
        line = lines.pop(0)
        tokens = line.split()
        assert tokens[0] == "binNumber"
        # _ = [int(x) for x in tokens[1:]]
     
        line = lines.pop(0)
        tokens = line.split()
        assert tokens[0] == "binClassNumber"
        # _ = [int(x) for x in tokens[1:]]
     
        line = lines.pop(0)
        tokens = line.split()
        assert tokens[0] == "binClass_widthVector"
        W = int(tokens[1])
     
        line = lines.pop(0)
        tokens = line.split()
        assert tokens[0] == "binClass_lengthVector"
        H = int(tokens[1])
     
        line = lines.pop(0)
        tokens = line.split()
        assert tokens[0] == "binClass_disponibility"
        # _ = [int(x) for x in tokens[1:]]

        try:
            line = lines.pop(0)
            tokens = line.split()
            assert tokens[0] == "itemClass_weight"
            p = [int(x) for x in tokens[1:]]
        except IndexError:
            p = [w[i]*h[i] for i in range(n)]

    return (n, H, W, [h[i]*d[i] for i in range(n)], w, [p[i]/h[i] for i in range(n)])  # vertical cut
    # return (n, W, H, [w[i]*d[i] for i in range(n)], h, [p[i]/w[i] for i in range(n)])  # horizontal cut -- as semifluid paper
