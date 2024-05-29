MAX_CELLS = 3500

SUM = 0
SUC = 1
ZERO = 2

SUC_SUM = SUM + SUC
ZERO_SUM = ZERO + SUM


class Port:
    def __init__(self, connected_cell = -1, connected_port = -1) -> None:
        self.connected_cell = connected_cell
        self.connected_port = connected_port

    def __str__(self) -> str:
        return f"(conn_cell: {self.connected_cell}, conn_port: {self.connected_port})"

class Cell:
    cell_counter = 0
    def __init__(self, cell_type, num_aux_ports):
        self.cell_id = Cell.cell_counter
        self.type = cell_type
        self.num_aux_ports = num_aux_ports
        self.ports = [Port() for _ in range(num_aux_ports + 1)] # + 1 for the main port
        Cell.cell_counter += 1
    
    def __str__(self):
        port_str = ','.join(str(port) for port in self.ports)
        return f"Cell id: {self.cell_id}\nType: {self.type}\nNum aux ports: {self.num_aux_ports}\nPorts: [{port_str}]"
        

def add_to_net(net, cell):
    net[cell.cell_id] = cell

def zero_cell(net):
    c = Cell(ZERO, 0)
    add_to_net(net, c)
    return c

def suc_cell(net):
    c = Cell(SUC, 1)
    add_to_net(net, c)
    return c

def sum_cell(net):
    c = Cell(SUM, 2)
    add_to_net(net, c)
    return c 

def link(cells, a: int, a_idx: int, b: int, b_idx: int):
    a_cell = cells[a]
    b_cell = cells[b]
    if(a_cell == -1 and b_cell != -1):
        b_cell.ports[b_idx] = -1
    elif (a_cell != -1 and b_cell == -1):
        a_cell.ports[a_idx] = -1
    else:
        a_cell.ports[a_idx] = Port(b, b_idx)
        b_cell.ports[b_idx] = Port(a, a_idx)

def delete_cell(cells, cell_id):
    cells[cell_id] = -1

def suc_sum(cells, suc, s):
    new_suc = suc_cell(cells)
    cells[new_suc.cell_id] = new_suc

    suc_principal = suc.ports[0]
    sum_principal = suc.ports[0]
    suc_first_aux = suc.ports[1]

    # link sum cell to what was connected to aux port of old suc
    link(cells, s.cell_id, 0, suc_first_aux.connected_cell, suc_first_aux.connected_port)
    # connect new suc principal port to whatever was connected in sum 2nd aux port
    # print(s)
    link(cells, new_suc.cell_id, 0, s.ports[2].connected_cell, s.ports[2].connected_port)
    # connect new suc aux port to sum 2nd aux port
    link(cells, new_suc.cell_id, 1, s.cell_id, 2)
    delete_cell(cells, suc.cell_id)

def zero_sum(cells, zero, s):
    link(cells, s.ports[1].connected_cell, s.ports[1].connected_port, s.ports[2].connected_cell, s.ports[2].connected_port)
    delete_cell(cells, zero.cell_id)
    delete_cell(cells, s.cell_id)

def check_rule(cell_a, cell_b):
    rule = cell_a.type + cell_b.type
    if rule == SUC_SUM:
        return (suc_sum, cell_a.cell_id, cell_b.cell_id)
    elif rule == ZERO_SUM:
        return (zero_sum, cell_a.cell_id, cell_b.cell_id)

def find_reducible(cells):
    for cell in cells:
        if(cell == -1):
            continue

        main_port = cell.ports[0]
        if main_port.connected_port == 0:
            return check_rule(cell, cells[main_port.connected_cell])
        
def church_encode(net, num: int) -> list:
    zero = zero_cell(net)

    to_connect_cell = zero
    to_connect_port = 0
    for _ in range(num):
        suc = suc_cell(net)
        link(net, suc.cell_id, 1, to_connect_cell.cell_id, to_connect_port)
        to_connect_cell = suc
        to_connect_port = 0
    return to_connect_cell.cell_id

def find_zero_cell(net):
    for c in net:
        if(c != -1):
            if c.type == ZERO:
                return c
    return None

def church_decode(net) -> int:
    cell = find_zero_cell(net)
    if not cell:
        print("Not a church encoded number net!")
    val = 0
    port = cell.ports[0]

    while (port != -1):
        port = net[port.connected_cell].ports[0]
        val += 1
    return val

def main():
    net = [-1 for _ in range(MAX_CELLS)]
    last_cell_id = church_encode(net, 1)
    last_id = church_encode(net, 1)
    s = sum_cell(net)

    # link right to main port of sum
    link(net, s.cell_id, 0, last_cell_id, 0)
    # link left to first aux port of sum
    link(net, s.cell_id, 1, last_id, 0)

    (reduce_function, a_id, b_id) = find_reducible(net)
    # print(reduce_function, a_id, b_id)
    reduce_function(net, net[a_id], net[b_id])

    (r, a, b) = find_reducible(net)
    r(net, net[a], net[b])

    val = church_decode(net)
    print(val)
    
    # print("\n".join(str(cell) for cell in net if cell != -1))

    # net = [-1 for _ in range(MAX_CELLS)]
    # z = zero_cell(net)
    # z_2 = zero_cell(net)

    # suc = suc_cell(net)

    # s = sum_cell(net)

    # # print(cells)
    # link(net, z.cell_id, 0, s.cell_id, 0)
    # link(net, z_2.cell_id, 0, s.cell_id, 2)
    # link(net, suc.cell_id, 1, s.cell_id, 1)

    # a = find_reducible(net)

    # a(net, z, s)

    
    


if __name__ == '__main__':
    main()
