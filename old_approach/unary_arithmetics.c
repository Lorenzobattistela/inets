#include "unary_arithmetics.h"

void mul_suc(Net *net, Cell *mul, Cell *suc, Connection *conn) {
  if (mul->symbol != MUL || suc->symbol != SUC || mul->deleted ||
      suc->deleted) {
    return;
  }
  // this rule is a little more complex. We will end up with a mul port, a sum
  // port and a dup port
  // first, we connect mul princ port to suc aux port conn
  // then we create a sum and a dup port
  // connect the sum princ port to the 2 aux port of mul
  // connect dup main port to whatever mul 1st aux port was connected to
  // connect the dup 1 aux port to 1 aux port of mul
  // connected the dup 2 aux port to 1 aux port of sum
  // delete suc
  connect(mul->principal_port, suc->auxiliary_ports[0]->connected_to);
  Cell *sum = sum_cell(net);
  Cell *dup = dup_cell(net);

  connect(sum->auxiliary_ports[1], mul->auxiliary_ports[1]->connected_to);
  connect(sum->principal_port, mul->auxiliary_ports[1]);
  connect(dup->principal_port, mul->auxiliary_ports[0]->connected_to);
  connect(mul->auxiliary_ports[0], dup->auxiliary_ports[0]);
  connect(dup->auxiliary_ports[1], sum->auxiliary_ports[0]);

  delete_cell(net, suc);

  conn->used = true;
}

void mul_zero(Net *net, Cell *mul, Cell *zero, Connection *conn) {
  if (mul->symbol != MUL || zero->symbol != ZERO || mul->deleted ||
      zero->deleted) {
    return;
  }
  // mul and zero interaction basically connects the first
  // aux port of mul to an erasor and the second one to zero
  Cell *erasor = erasor_cell(net);
  connect(mul->auxiliary_ports[0]->connected_to, erasor->principal_port);
  connect(mul->auxiliary_ports[1]->connected_to, zero->principal_port);
  delete_cell(net, mul);

  conn->used = true;
  return;
}

void zero_erasor(Net *net, Cell *zero, Cell *erasor, Connection *conn) {
  if (zero->symbol != SUC || erasor->symbol != ERA || zero->deleted ||
      erasor->deleted) {
    return;
  }
  // simply delete both cells, nothing remains
  delete_cell(net, zero);
  delete_cell(net, erasor);

  conn->used = true;
}

void suc_erasor(Net *net, Cell *suc, Cell *erasor, Connection *conn) {
  if (suc->symbol != SUC || erasor->symbol != ERA || suc->deleted ||
      erasor->deleted) {
    return;
  }

  // erase a suc simply deletes the suc cell and connects the erasor to whatever
  // suc aux port was connected to
  connect(erasor->principal_port, suc->auxiliary_ports[0]->connected_to);
  delete_cell(net, suc);

  conn->used = true;
}

void zero_dup(Net *net, Cell *zero, Cell *dup, Connection *conn) {
  if (zero->symbol != ZERO || dup->symbol != DUP || zero->deleted ||
      dup->deleted) {
    return;
  }
  // zero main to dup main should clone 0 and connect it both
  // to whatever was connected to dup aux nodes
  Cell *new_zero = zero_cell(net);
  connect(zero->principal_port, dup->auxiliary_ports[0]->connected_to);
  connect(new_zero->principal_port, dup->auxiliary_ports[1]->connected_to);
  delete_cell(net, dup);

  conn->used = true;
  return;
}

void suc_dup(Net *net, Cell *suc, Cell *dup, Connection *conn) {
  if (suc->symbol != SUC || dup->symbol != DUP || suc->deleted ||
      dup->deleted) {
    return;
  }
  // if a suc main port connects to a dup main port, simply return
  // two sucs with main ports connected to the aux ports of dup
  // therefore we create a new suc and connect the old one
  Cell *new_suc = suc_cell(net);
  // connect dup principal port with suc aux conn
  connect(dup->principal_port, suc->auxiliary_ports[0]->connected_to);
  // conn suc principal_port to dup aux conn
  connect(suc->principal_port, dup->auxiliary_ports[0]->connected_to);
  connect(new_suc->principal_port, dup->auxiliary_ports[1]->connected_to);

  // connect suc and new suc aux ports with dup aux ports
  connect(suc->auxiliary_ports[0], dup->auxiliary_ports[0]);
  connect(new_suc->auxiliary_ports[0], dup->auxiliary_ports[1]);

  conn->used = true;
  return;
}

void suc_sum(Net *net, Cell *suc, Cell *sum, Connection *conn) {
  if (suc->symbol != SUC || sum->symbol != SUM || suc->deleted ||
      sum->deleted) {
    return;
  }
  // a suc port connected with a + port
  // makes the second aux port connected to a suc cell intermediately
  // and connects the + main port to whatever the suc aux port was connected
  Cell *new_suc = suc_cell(net);

  // first connect sum main port to s aux port conn
  connect(sum->principal_port, suc->auxiliary_ports[0]->connected_to);
  // conn the new suc main port to sum aux connected to
  connect(new_suc->principal_port, sum->auxiliary_ports[1]->connected_to);
  // conn the new suc auxiliar to the + second aux
  connect(new_suc->auxiliary_ports[0], sum->auxiliary_ports[1]);

  // now simply delete the old suc
  delete_cell(net, suc);

  // finally remove the connection from the array (maybe marking as used)
  conn->used = true;
}

void zero_sum(Net *net, Cell *zero, Cell *sum, Connection *conn) {
  if (zero->symbol != ZERO || sum->symbol != SUM || zero->deleted ||
      sum->deleted) {
    return;
  }
  // for this rule, we simply connect whatever is connected to 1st aux
  // port of the sum rule to whatever in the 2nd aux
  connect(sum->auxiliary_ports[0]->connected_to,
          sum->auxiliary_ports[1]->connected_to);

  // and delete both + and 0
  delete_cell(net, zero);
  delete_cell(net, sum);

  conn->used = true;
}
