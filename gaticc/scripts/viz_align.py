import math
import argparse

input_dims = [5, 3, 5]
sa_arch = [9, 8, 8]
dram_width = 32

def ceil_mod(a, b):
    return int(math.ceil(a / b) * b)

def ceil_div(a, b):
    return math.ceil(a / b)

def align_dims(input_dims, sa_arch):
    new_dims = [ceil_mod(input_dims[0], sa_arch[2]), input_dims[1], input_dims[2]]
    return new_dims

def align_pw(dims, sa_arch):
    new_dims = [ceil_mod(dims[0], sa_arch[1]), ceil_mod(dims[1], sa_arch[0]), dims[2], dims[3]]
    return new_dims

def print_tbl(table_data):
    for i in table_data:
        for j in i:
            print(f"{j}\t", end='')
        print()

def viz_sa_input(input_dims, sa_arch, dram_width):
    dk = dram_width // sa_arch[2]
    new_dims = align_dims(input_dims, sa_arch)
    print(f"input_dims: {input_dims}, sa_arch: {sa_arch}, dram_width: {dram_width} new_dims: {new_dims}")
    table_data = []
    for c in range(new_dims[0] // sa_arch[2]):
        for e in range(ceil_mod(new_dims[1] * new_dims[2], dk) // dk):
            row = []
            for ci in range(sa_arch[2]):
                for ei in range(dk):
                    chan = c * sa_arch[2] + ci
                    elem = e * dk + ei
                    if chan >= input_dims[0] or elem >= input_dims[1] * input_dims[2]:
                        row.append("0")
                    else:
                        row.append(f"c{chan}e{elem}")
            table_data.append(row)
    return table_data

def align_dim_fc_bias(input_dims, vasize):
    return [ceil_mod(input_dims[0], vasize)] 

# independent of DRAM Width
def viz_fc_bias(input_dims, sa_arch, vasize):
    tail_blocks = sa_arch[2] if sa_arch is not None and len(sa_arch) == 3 else vasize
    dk = vasize // tail_blocks # N_SA
    aligned_dims = align_dim_fc_bias(input_dims, vasize)
    iterations = aligned_dims[0] // (tail_blocks * dk)
    table_data = []
    for i in range(iterations):
        for j in range(dk):
            row = []
            for k in range(tail_blocks):
                index = i * tail_blocks * dk + j + k * dk
                if index >= input_dims[0]:
                    row.append("0")
                else:
                    row.append(f"e{index}")
            table_data.append(row)    
    return table_data


# align weights for a pointwise convolution
def viz_sa_pointwise(dims, sa_arch):
    aligned_dims = align_pw(dims, sa_arch)
    kern_itr = ceil_div(aligned_dims[0], sa_arch[0])
    chan_itr = ceil_div(aligned_dims[1], sa_arch[1])

    table_data = []
    for ki in range(kern_itr):
        for ci in range(chan_itr):
            for c in reversed(range(sa_arch[1])):
                row = []
                for r in range(sa_arch[0]):
                    row.append(f"k{ki*sa_arch[2]+r}c{(ci*sa_arch[0]+c)}")
                table_data.append(row)
    return table_data

def main():
    parser = argparse.ArgumentParser(description="Visualize aligned dimensions for a given input")
    parser.add_argument('--sa_input', action="store_true", help="Visualize SA Input")
    parser.add_argument('--fc_bias', action="store_true", help="Visualize FC Bias")
    parser.add_argument('--sa_pointwise', action="store_true", help="Visualize SA Pointwise Alignment")
    parser.add_argument('--input_dims', type=int, nargs='+', help="Input dimensions [depth, height, width] or [N]")
    parser.add_argument('--sa_arch', type=int, nargs=3, help="Systolic array arch dimensions [rows, cols, N]")
    parser.add_argument('--vasize', type=int, help="Vector Array Dimensions [int]")
    parser.add_argument('--dram_width', type=int, help="DRAM width")
    args = parser.parse_args()
    if args.sa_input:
        table_data = viz_sa_input(args.input_dims, args.sa_arch, args.dram_width)
    elif args.fc_bias:
        table_data = viz_fc_bias(args.input_dims, args.sa_arch, args.vasize)
    elif args.sa_pointwise:
        table_data = viz_sa_pointwise(args.input_dims, args.sa_arch)
    print_tbl(table_data)

if __name__ == "__main__":
    main()
