# Problem Petani - Ayam - Jagung - Rubah dengan DFS

initial_state = ('L', 'L', 'L', 'L')
goal_state = ('R', 'R', 'R', 'R')

def is_valid(state):
    farmer, chicken, corn, fox = state
    
    # Jika petani tidak ada di sisi yang sama:
    # Ayam makan jagung
    if farmer != chicken and chicken == corn:
        return False
    # Rubah makan ayam
    if farmer != chicken and chicken == fox:
        return False
    return True

def get_successors(state):
    successors = []
    farmer, chicken, corn, fox = state
    
    # Semua kemungkinan perpindahan:
    candidates = [('farmer',), ('chicken',), ('corn',), ('fox',)]
    
    for move in candidates:
        new_state = list(state)
        # Pindahkan petani
        new_state[0] = 'R' if farmer == 'L' else 'L'
        
        # Jika membawa sesuatu
        if len(move) == 1 and move[0] != 'farmer':
            idx = ['farmer', 'chicken', 'corn', 'fox'].index(move[0])
            if state[idx] == farmer:  # hanya bisa dipindah kalau satu sisi
                new_state[idx] = new_state[0]
            else:
                continue
        
        new_state = tuple(new_state)
        if is_valid(new_state):
            successors.append(new_state)
    
    return successors

def dfs():
    stack = [(initial_state, [initial_state])]
    visited = set()
    
    while stack:
        (state, path) = stack.pop()
        if state in visited:
            continue
        visited.add(state)
        
        if state == goal_state:
            return path
        
        for succ in get_successors(state):
            stack.append((succ, path + [succ]))
    return None

if __name__ == "__main__":
    solution = dfs()
    if solution:
        print("Solusi ditemukan:")
        for step in solution:
            print(step)
    else:
        print("Tidak ada solusi ditemukan.")