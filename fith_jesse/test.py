import numpy as np
from time import time
from argparse import ArgumentParser

def main(n_guesses, n_holes):
    transfermatrix = np.array([
        [ 0, 0.5,   0,   0, 0],
        [ 1,   0, 0.5,   0, 0],
        [ 0, 0.5,   0, 0.5, 0],
        [ 0,   0, 0.5,   0, 1],
        [ 0,   0,   0, 0.5, 0]])
    initial_state = np.ones(n_holes)/n_holes 
    # print(initial_state)

    # choices = [choose(i, n_holes) for i in range(n_holes)]
    choices = None
    avgs = {}
    i = 0
    starttime1 = time()
    for guesses in configurations(n_guesses, n_holes):
        # print(guesses)
        # starttime = time()
        avg = run_guesses(initial_state, transfermatrix, guesses, n_guesses, choices)
        # runtime = time()-starttime
        if i%10000 == 0:
            print(f"Strategy: {i}", end="\r")
        # print(f"Strategy: {i}, average number of guesses {avg:.2f}", end="\r")
        # avgs[str(guesses)] = avg
        # avgs[guesses] = avg
        avgs[i] = avg
        i += 1
    print(end="                                                                                                                   \r")
    runtime1 = time()-starttime1
    # print(avgs)
    # print(f"{n_guesses}: {min(avgs.values()):.2f} {min(avgs, key=avgs.get)}, {runtime1:.4f}, {i}                                                                                                             ")
    min_performance = np.inf
    min_configuration = []
    for i, configuration in enumerate(configurations(n_guesses, n_holes)):
        if avgs[i] < min_performance:
            min_performance = avgs[i]
            min_configuration = configuration
    print(f"{n_guesses}: {min_performance:.2f} {min_configuration}, {runtime1:.4f}, {i}                                                                                                             ")
    # print(min(avgs))

def choose(hole, n_holes):
    measurement = np.eye(n_holes,n_holes)
    measurement[hole][hole] = 0
    # print(measurement)
    return measurement

def configurations(N, n_holes):
    # print(n)
    if N == 1:
        for i in range(n_holes):
            yield [i]
    else:
        for configuration in configurations(N-1, n_holes):
            for i in range(n_holes):
                yield np.concatenate(([i], configuration))    

def run_guesses(initial_state, transfermatrix, guesses, n_guesses, choices = None):
    state = initial_state.copy()
    avg = 0
    for n, guess in enumerate(guesses):
        avg += state[guess]*(n+1)
        # state = np.inner(choose(guess, n_holes), state)
        # state = np.inner(choices[guess], state)
        state[guess] = 0
        state = np.inner(transfermatrix,state)
        # state = state/np.sum(state) # NORMALISE (Not the correct calculation)
        # print(f"{n+1} {avg:.2f} {state} {np.sum(state)}")
    avg += (n_guesses+1)*np.sum(state)
    return avg

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("-N", "--n_guesses", default=6, type=int, help="Number of guesses")
    argparser.add_argument("-H", "--n_holes", default=5, type=int, help="Number of holes")
    args = argparser.parse_args()

    N = args.n_guesses
    n_holes = args.n_holes
    main(N, n_holes)
    # for i in range(1, N+1):
    #     main(i, n_holes)
    # transfermatrix = np.array([
    #     [ 0, 0.5,   0,   0, 0],
    #     [ 1,   0, 0.5,   0, 0],
    #     [ 0, 0.5,   0, 0.5, 0],
    #     [ 0,   0, 0.5,   0, 1],
    #     [ 0,   0,   0, 0.5, 0]])
    # initial_state = np.ones(n_holes)/n_holes

    # print(run_guesses(initial_state, transfermatrix, [1, 1, 3, 3, 1, 1], 6))
    # print(run_guesses(initial_state, transfermatrix, [2], 1))