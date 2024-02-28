# Made by Onur Danaci
# Edits by Hans Norg
import numpy as np
from numba import njit


dtype = np.complex64

bs0 = np.array([[1],[0]],dtype = dtype)

bs1 = np.array([[0],[1]],dtype = dtype)


P0 = bs0 @ bs0.conj().T

P1 = bs1 @ bs1.conj().T


sx = np.array([[0,1],[1,0]], dtype = dtype)
sy = np.array([[0,-1j],[1j,0]], dtype = dtype)
sz = np.array([[1,0],[0,-1]], dtype = dtype)

Id = np.array([[1,0],[0,1]], dtype=dtype)

GrSt = (Id + sz)/2

ExSt = (Id - sz)/2

h = (1/np.sqrt(2)) * np.array([[1,1],[1,-1]], dtype=dtype)

s = np.array([[1,0],[0,1j]], dtype=dtype)

t = np.array([[1,0],[0,(1+1j)/np.sqrt(2)]], dtype=dtype)




rx = lambda x: np.array([[np.cos(x/2),-1j*np.sin(x/2)],[-1j*np.sin(x/2),np.cos(x/2)]], dtype=dtype)

ry = lambda x: np.array([[np.cos(x/2),-np.sin(x/2)],[np.sin(x/2),np.cos(x/2)]], dtype=dtype)

rz = lambda x: np.array([[np.exp(-1j*x/2),0],[0,np.exp(+1j*x/2)]], dtype=dtype)

# def get_Ising_h_term(h, n_qubits):
#     M = len(h)
    
#     H = 0
#     for ih in range(M):
#         H += h[ih]*Z(ih, n_qubits)
        
#     return H

# def get_Ising_J_term(J, n_qubits):
#     M = len(J)
    
#     H = 0
#     for ij in range(0,M,2):
#         H += J[ih]*ZZ(ij, n_qubits)
        
#     return H

# def get_Ising(h, j, n_qubits):
#     return get_Ising_h_term(h, n_qubits) + get_Ising_J_term(J, n_qubits)
    
@njit
def from_string_test(string):
    # U = np.array([[1, 0], [0, 1]], dtype=dtype)
    for ind, el in enumerate(string):
        if el == "I":
            op = Id
        elif el == "X":
            op = sx
        elif el == "Y":
            op = sy
        elif el == "Z":
            op = sz
        else:
            raise ValueError("Pauli identifier " + el + " not found")            
        if ind == 0:
            U = op.copy()
        else:
            U = np.kron(U, op)
    return U

def from_string(string):
    U = np.array([1], dtype = dtype)
    for el in string:
        if el == "I":
            U = np.kron(U, Id)
        elif el == "X":
            U = np.kron(U, sx)
        elif el == "Y":
            U = np.kron(U, sy)
        elif el == "Z":
            U = np.kron(U, sz)
        else:
            raise ValueError("Pauli identifier " + el + " not found")
            
    return U

def ZZ(targ_q, n_qubits = 4):
    U = np.array([1], dtype = dtype)
    for iq in range(n_qubits):
        if iq == targ_q:
            U = np.kron(U, sz)
        elif iq == targ_q + 1:
            U = np.kron(U,sz)
        else:
            U = np.kron(U, Id)
            
    return U

def YY(targ_q, n_qubits = 4):
    U = np.array([1], dtype = dtype)
    for iq in range(n_qubits):
        if iq == targ_q:
            U = np.kron(U, sy)
        elif iq == targ_q + 1:
            U = np.kron(U,sy)
        else:
            U = np.kron(U, Id)
            
    return U

def XX(targ_q, n_qubits = 4):
    U = np.array([1], dtype = dtype)
    for iq in range(n_qubits):
        if iq == targ_q:
            U = np.kron(U, sx)
        elif iq == targ_q + 1:
            U = np.kron(U,sx)
        else:
            U = np.kron(U, Id)
            
    return U

def X(targ_q, n_qubits = 4):
    U = np.array([1], dtype = dtype)
    
    for iq in range(n_qubits):
        if iq == targ_q:
            U = np.kron(U, sx)
        else:
            U = np.kron(U, Id)
            
    return U

def Y(targ_q, n_qubits = 4):
    U = np.array([1],dtype = dtype)
    
    for iq in range(n_qubits):
        if iq == targ_q:
            U = np.kron(U, sy)
        else:
            U = np.kron(U, Id)
            
    return U

def Z(targ_q, n_qubits = 4):
    U = np.array([1],dtype = dtype)
    
    for iq in range(n_qubits):
        if iq == targ_q:
            U = np.kron(U, sz)
        else:
            U = np.kron(U, Id)
            
    return U

def I(targ_q, n_qubits = 4):
    U = np.array([1],dtype = dtype)
    
    for iq in range(n_qubits):
        if iq == targ_q:
            U = np.kron(U, Id)
        else:
            U = np.kron(U, Id)
            
    return U

def H(targ_q, n_qubits = 4):
    U = np.array([1],dtype = dtype)
    
    for iq in range(n_qubits):
        if iq == targ_q:
            U = np.kron(U, h)
        else:
            U = np.kron(U, Id)
            
    return U

def S(targ_q, n_qubits = 4):
    U = np.array([1],dtype = dtype)
    
    for iq in range(n_qubits):
        if iq == targ_q:
            U = np.kron(U, s)
        else:
            U = np.kron(U, Id)
            
    return U

def T(targ_q, n_qubits = 4):
    U = np.array([1],dtype = dtype)
    
    for iq in range(n_qubits):
        if iq == targ_q:
            U = np.kron(U, t)
        else:
            U = np.kron(U, Id)
            
    return U

def Rx(targ_q, rads, n_qubits = 4):
    U = np.array([1],dtype = dtype)
    
    for iq in range(n_qubits):
        if iq == targ_q:
            U = np.kron(U, rx(rads))
        else:
            U = np.kron(U, Id)
            
    return U

def Ry(targ_q, rads, n_qubits = 4):
    U = np.array([1],dtype = dtype)
    
    for iq in range(n_qubits):
        if iq == targ_q:
            U = np.kron(U, ry(rads))
        else:
            U = np.kron(U, Id)
            
    return U

def Rz(targ_q, rads, n_qubits = 4):
    U = np.array([1],dtype = dtype)
    
    for iq in range(n_qubits):
        if iq == targ_q:
            U = np.kron(U, rz(rads))
        else:
            U = np.kron(U, Id)
            
    return U
        
def CX(k,l,n_qubits = 4):
    
    ctr_op = np.array([1],dtype = dtype)
    
    not_op = ctr_op
    
    for iq in range(n_qubits):
        if iq == k:
            ctr_op = np.kron(ctr_op, GrSt)
            not_op = np.kron(not_op, ExSt)
        elif iq ==l:
            ctr_op = np.kron(ctr_op, Id)
            not_op = np.kron(not_op, sx)
        else:
            ctr_op = np.kron(ctr_op, Id)
            not_op = np.kron(not_op, Id)
            

    return ctr_op + not_op


def CZ(k,l,n_qubits = 4):
    h_ =  np.array([1],dtype = dtype)
    
    for iq in range(n_qubits):
        if iq ==l:
            h_ = np.kron(h_,h)
        else:
            h_ = np.kron(h_, Id)
    
    return h_ @ CX(k,l, n_qubits) @ h_
    
def unitary_init(n_qubits = 4):
    U = Id
    
    for iq in range(1,n_qubits):
        U = np.kron(U, Id)
        
    return U

def unitary_init(n_qubits = 4):
    U = Id
    
    for iq in range(1,n_qubits):
        U = np.kron(U, Id)
        
    return U

def state_initializer(n_qubits = 4):
    state = np.array([1],dtype = dtype)
    
    for iq in range(n_qubits):
        state = np.kron(state, bs0)
        
    return state

def ket2dm(state):
    
    return state @ state.conj().T

def dm_initializer(n_qubits = 4):
    
    state = np.array([1],dtype = dtype)
    
    for iq in range(n_qubits):
        state = np.kron(state, P0)
        
    return state

def apply_unitary(unitary, rho):
    
    return unitary @ rho @ unitary.conj().T

def apply_cptp(K_list, rho):
    
    rho_t = np.zeros_like(rho, dtype = dtype)
    
    for K in K_list:
        
        rho_t += K@ rho @ K.conj().T 
        
    return rho_t






# def depolarizing_fn(error_prob):
#         PId, Px, Py, Pz = 1 - 3*error_prob/4, error_prob/4, error_prob/4, error_prob/4
        
#         probabilities = [PId, Px, Py, Pz]
        
#         unitaries = [I, X, Y, Z]
        
#         K_list_fn = []
        
#         for iK in range(len(unitaries)):
#             def K_fun(targ_qub, n_qubits): 
#                 return np.sqrt(probabilities[iK])*unitaries[iK](targ_qub, n_qubits)
            
#             K_list_fn.append(K_fun)
            
#         return K_list_fn


if __name__ == "__main__":
    from time import time
    for string in ["XIZY", "ZYZIZYXXZY", "IZYXYZYXZYZI"]:
        print()
        print(string)
        starttime = time()
        original = from_string(string)
        print("original", time()-starttime)
        starttime = time()
        test = from_string_test(string)
        print("test    ", time()-starttime)
        print(original.all()==test.all())
        N = 10
        starttime = time()
        for i in range(N):
            from_string(string)
        print("original", time()-starttime)
        starttime = time()
        for i in range(N):
            from_string_test(string)
        print("test    ", time()-starttime)