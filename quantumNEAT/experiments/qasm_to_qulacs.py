# By Hans Norg: Modification of convert_QASM_to_qulacs_circuit from qulacs.converter.qasm_converter
# from qulacs.converter.qasm_converter import convert_QASM_to_qulacs_circuit

import re
import typing
from typing import List

import numpy as np
from qulacs import ParametricQuantumCircuit
from qulacs.gate import DepolarizingNoise, TwoQubitDepolarizingNoise

FIXED_POINT_PATTERN = r"[+-]?\d+(?:\.\d*)?|\.\d+"
FLOATING_POINT_PATTERN = r"[eE][-+]?\d+"
GENERAL_NUMBER_PATTERN = (
    rf"(?:{FIXED_POINT_PATTERN})(?:{FLOATING_POINT_PATTERN})?"  # noqa
)

def convert_QASM_to_qulacs_circuit(
    input_strs: typing.List[str], *, remap_remove: bool = False, 
    phys_noise: bool = False, depol_prob:float = None,
) -> ParametricQuantumCircuit:
    # convert QASM List[str] to qulacs QuantumCircuit.
    # constraints: qreg must be named q, and creg cannot be used.

    mapping: List[int] = []

    for instr_moto in input_strs:
        # process input string for parsing instruction.
        instr = instr_moto.lower().strip().replace(" ", "").replace("\t", "")
        if instr == "":
            continue
        # print(instr)
        if instr[0:4] == "qreg":
            matchobj = re.match(r"qregq\[(\d+)\];", instr)
            assert matchobj is not None
            ary = matchobj.groups()
            cir = ParametricQuantumCircuit(int(ary[0]))
            if len(mapping) == 0:
                mapping = list(range(int(ary[0])))
        elif instr[0:2] == "cx":
            matchobj = re.match(r"cxq\[(\d+)\],q\[(\d+)\];", instr)
            assert matchobj is not None
            ary = matchobj.groups()
            cir.add_CNOT_gate(mapping[int(ary[0])], mapping[int(ary[1])])
            if phys_noise:
                cir.add_gate(TwoQubitDepolarizingNoise(mapping[int(ary[0])], mapping[int(ary[1])], depol_prob))
        elif instr[0:2] == "cz":
            matchobj = re.match(r"czq\[(\d+)\],q\[(\d+)\];", instr)
            assert matchobj is not None
            ary = matchobj.groups()
            cir.add_CZ_gate(mapping[int(ary[0])], mapping[int(ary[1])])
            if phys_noise:
                cir.add_gate(TwoQubitDepolarizingNoise(mapping[int(ary[0])], mapping[int(ary[1])], depol_prob))
        elif instr[0:2] == "rx":
            if "pi/2" in instr:
                matchobj = re.match(rf"rx\(([+-]?)pi/2\)q\[(\d+)\];", instr)
                # print(matchobj)
                assert matchobj is not None
                ary = matchobj.groups()
                # print(ary)
                cir.add_RX_gate(mapping[int(ary[1])], -float(ary[0]+str(np.pi/2)))
            elif "pi" in instr:
                matchobj = re.match(rf"rx\(([+-]?)pi\)q\[(\d+)\];", instr)
                # print(matchobj)
                assert matchobj is not None
                ary = matchobj.groups()
                # print(ary)
                cir.add_RX_gate(mapping[int(ary[1])], -float(ary[0]+str(np.pi)))
            else:
                matchobj = re.match(rf"rx\(({GENERAL_NUMBER_PATTERN})\)q\[(\d+)\];", instr)
                assert matchobj is not None
                ary = matchobj.groups()
                cir.add_parametric_RX_gate(mapping[int(ary[1])], -float(ary[0]))
            if phys_noise:
                cir.add_gate(DepolarizingNoise(mapping[int(ary[1])], depol_prob))
        elif instr[0:2] == "ry":
            if "pi/2" in instr:
                matchobj = re.match(rf"ry\(([+-]?)pi/2\)q\[(\d+)\];", instr)
                # print(matchobj)
                assert matchobj is not None
                ary = matchobj.groups()
                # print(ary)
                cir.add_RY_gate(mapping[int(ary[1])], -float(ary[0]+str(np.pi/2)))
            elif "pi" in instr:
                matchobj = re.match(rf"ry\(([+-]?)pi\)q\[(\d+)\];", instr)
                # print(matchobj)
                assert matchobj is not None
                ary = matchobj.groups()
                # print(ary)
                cir.add_RY_gate(mapping[int(ary[1])], -float(ary[0]+str(np.pi)))
            else:
                matchobj = re.match(rf"ry\(({GENERAL_NUMBER_PATTERN})\)q\[(\d+)\];", instr)
                assert matchobj is not None
                ary = matchobj.groups()
                cir.add_parametric_RY_gate(mapping[int(ary[1])], -float(ary[0]))
            if phys_noise:
                cir.add_gate(DepolarizingNoise(mapping[int(ary[1])], depol_prob))
        elif instr[0:2] == "rz":
            if "pi/2" in instr:
                matchobj = re.match(rf"rz\(([+-]?)pi/2\)q\[(\d+)\];", instr)
                # print(matchobj)
                assert matchobj is not None
                ary = matchobj.groups()
                # print(ary)
                cir.add_RZ_gate(mapping[int(ary[1])], -float(ary[0]+str(np.pi/2)))
            elif "pi" in instr:
                matchobj = re.match(rf"rz\(([+-]?)pi\)q\[(\d+)\];", instr)
                # print(matchobj)
                assert matchobj is not None
                ary = matchobj.groups()
                # print(ary)
                cir.add_RZ_gate(mapping[int(ary[1])], -float(ary[0]+str(np.pi)))
            else:
                matchobj = re.match(rf"rz\(({GENERAL_NUMBER_PATTERN})\)q\[(\d+)\];", instr)
                assert matchobj is not None
                ary = matchobj.groups()
                cir.add_parametric_RZ_gate(mapping[int(ary[1])], -float(ary[0]))
            if phys_noise:
                cir.add_gate(DepolarizingNoise(mapping[int(ary[1])], depol_prob))
        elif remap_remove and instr[0:4] == "//q[":
            matchobj = re.match(r"//q\[(\d+)-->q\[(\d+)\]", instr)
            assert matchobj is not None
            ary = matchobj.groups()
            if not (ary is None):
                mapping[int(ary[0])] = int(ary[1])
        elif remap_remove and instr[0:8] == "//qubits":
            matchobj = re.match(r"//qubits:(\d+)", instr)
            assert matchobj is not None
            ary = matchobj.groups()
            mapping = list(range(int(ary[0])))
        elif instr == "openqasm2.0;" or instr == 'include"qelib1.inc";':
            # related to file format, not for read.
            pass
        else:
            raise RuntimeError(f"unknown line: {instr}")
        # print("next")
    return cir
