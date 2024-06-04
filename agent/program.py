# COMP30024 Artificial Intelligence, Semester 1 2024
# Project Part B: Game Playing Agent

import random, math, time
import numpy as np
from referee.game import PlayerColor, Action, PlaceAction, Coord, constants, Direction, BOARD_N, GameEnd
from referee.game.pieces import _TEMPLATES, Piece
from typing import Dict, List


def get_colour_coordinates(board_State: Dict[Coord, PlayerColor], player_colour:PlayerColor):
    """
    This function iterates through the tokens on the board, appending token 
    coordinates to a list if they match the colour specified by the argument. 
    Returns: list of coordinates occupied by a player colour,  List[Coord].
    """
    return [coord for coord, colour in board_State.items() if colour == player_colour]

def remove_full_rows_and_columns(board_state: Dict[Coord, PlayerColor]):
    """
    Check for completed full rows and columns on the board, and remove them.
    
    Args:
        board_state (dict): A dictionary mapping coordinates to player colors.
        
    Returns:
        dict: The updated board state with completed full rows and columns removed.
    """
    rows, cols = BOARD_N, BOARD_N  
    completed_rows = []
    completed_cols = []
    
    # Check for completed rows
    for row in range(rows):
        row_cells = [board_state.get(Coord(row, col)) for col in range(cols)]
        if all(cell is not None for cell in row_cells):
            completed_rows.append(row)
    
    # Check for completed columns
    for col in range(cols):
        col_cells = [board_state.get(Coord(row, col)) for row in range(rows)]
        if all(cell is not None for cell in col_cells):
            completed_cols.append(col)

    # Find completed row tokens
    to_delete_rows = [Coord(row, col) for row in completed_rows for col in range(cols)]
    
    # Find completed column tokens
    to_delete_cols =  [Coord(row, col) for col in completed_cols for row in range(rows)]

    # Remove tokens from board
    remove = set (to_delete_cols + to_delete_rows)
    for item in remove:
            del board_state[item]
        
def get_adjacent_coords (token: Coord):
    """ Generate  adjacent coords around a given coord
    
    Args: A single Coord
    
    Return: return a list of Coords, adjacent to the given Coord
    """
    coord = Coord(token.r, token.c)

    return [coord.up(), coord.down(), coord.left(), coord.right()]

def get_random_square()->Coord:
    # Generate random row and column indices
    random_row = random.randint(0, BOARD_N - 1)
    random_col = random.randint(0, BOARD_N - 1)
    
    # Return the random square as a tuple of row and column indices
    return Coord(random_row, random_col)

def apply_ansi(
    text: str, 
    bold: bool = True, 
    color: str | None = None
):
    """
    Wraps some text with ANSI control codes to apply terminal-based formatting.
    Note: Not all terminals will be compatible!
    """
    bold_code = "\033[1m" if bold else ""
    color_code = ""
    if color == "r":
        color_code = "\033[31m"
    if color == "b":
        color_code = "\033[34m"
    return f"{bold_code}{color_code}{text}\033[0m"

def render_board(
    board: dict[Coord, PlayerColor], 
    target: Coord | None = None,
    ansi: bool = False
) -> str:
    """
    Visualise the Tetress board via a multiline ASCII string, including
    optional ANSI styling for terminals that support this.

    If a target coordinate is provided, the token at that location will be
    capitalised/highlighted.
    """
    output = ""
    for r in range(BOARD_N):
        for c in range(BOARD_N):
            if board.get(Coord(r, c), None):
                is_target = target is not None and Coord(r, c) == target
                color = board[Coord(r, c)]
                color = "r" if color == PlayerColor.RED else "b"
                text = f"{color}" if not is_target else f"{color.upper()}"
                if ansi:
                    output += apply_ansi(text, color=color, bold=is_target)
                else:
                    output += text
            else:
                output += "."
            output += " "
        output += "\n"
    return output


def identify_clusters(board, player_colour):
    """
    Identifies clusters of tokens belonging to a specified player colour with 
    at least one free adjacent cell and traces the edges

    Args:
        board (Dict[Coord, PlayerColor]): The current state of the game board.
        player_colour (PlayerColor): The colour of the player's tokens 

    Returns:
        List[Set[Coord]]: A list of sets, each containing coordinates 
        representing the edges of a cluster of tokens 
    """
    clusters = []
    visited = set()

    # Iterate through board to identify tokesn for teh cluster
    for coord,colour in board.items(): 
        if colour == player_colour and coord not in visited:
            visited.add(coord)

            # Do a depth-first search to ideantify tokens of the same cluster
            visited, edges = dfs(board, player_colour, coord, visited)
            clusters.append(edges)
    
    return clusters

def dfs(board, player_colour, start, visited):
    """
    Performs a depth-first search (DFS) algorithm to explore adjacent tokens to 
    identify which tokens on a sigular colour belong to which cluster

    Args:
        board (Dict[Coord, PlayerColor]): The current state of the game board
        player_colour (PlayerColor): The colour of the player's tokens
        start (Coord): Starting coordinate for DFS traversal
        visited (Set[Coord]): A set containing visited coordinates  

    Returns:
        Tuple[Set[Coord], Set[Coord]]: A tuple containing two sets:
            1. The first set is all coordinates visited during DFS 
            2. The second set is coordinates representing the clusters edges 
    """

    edges = set() # set for edges
    stack = [] # DFS traversal

    stack.append(start)
    visited.add(start)
    visited_clus = set()  # set for tracking visited coordinates
    visited_clus.add(start) 
    
    while stack:
        top_stack = stack.pop()
        adjacent = get_adjacent_coords(top_stack)

        for adj in adjacent:
            # Look at adjacent tokens
            if adj in board and board[adj]== player_colour and adj not in visited:
                stack.append(adj)
                visited.add(adj)
                visited_clus.add(adj)

            # Identify the edge tokens    
            elif adj not in visited or board.get(adj) != player_colour:
                edges.add(top_stack)

    return visited_clus, edges

def possible_moves(board, player_colour, turn_count, simulation_mode: bool):
    """
    Generates a list of possible moves for the player based on the current game 
    board state

    Args:
        board (dict): The current state of the game board
        player_colour (PlayerColor): The colour of the player making the moves
        turn_count (int): The current turn count in the game

    Returns:
        List[Piece]: A list of valid tetromino pieces representing possible 
        moves
    """
    cluster_list = identify_clusters(board, player_colour)
    

    valid_moves = []
    if simulation_mode == 1:
        while len(valid_moves) == 0:
            if len(cluster_list) != 0:
                cluster_idx = random.randrange(0,len(cluster_list))
            else:
                break
            cluster = cluster_list[cluster_idx]

            for coord in range (len(cluster)):
                if len(cluster) != 0:
                    coord = random.choice(list(cluster))
                else:
                    break  
                valid_moves.extend(
                    generate_valid_tetrinos(coord, board, player_colour,turn_count, 1))
                if len(valid_moves) != 0:
                    break
                cluster.remove(coord)
            cluster_list.pop(cluster_idx)
            
        if len(valid_moves) != 0:
            valid_move = random.randrange(0,len(valid_moves))
            valid_move = valid_moves[valid_move]
            valid_moves = []
            valid_moves.append(valid_move)
    else:
    # iterate through the edge tokens, usinh them to generate valid moves 
        for cluster in cluster_list:
            for coord in cluster:
                valid_moves.extend(
                    generate_valid_tetrinos(coord, board, player_colour,turn_count, 0))
    return valid_moves

def generate_valid_tetrinos(start_coord: Coord, board, player_colour, turn_count, sim_mode: bool) -> List[Piece]:
    
    """
    Generate all possible variations of tetromino pieces by placing them at 
    each adjacent square origin
    
    Args:
        start_coord (Coord): Starting coordinate for generating adjacent pieces
        board (Dict[Coord, PlayerColor]) : The game board
        player_colour (PlayerColour): The color of the player's pieces
        turn_count (int): The current turn count

    Returns:
        List[Piece]: A list of valid tetromino pieces placed at each adjacent 
        square origin
    """
    valid_adjacent_pieces = []

    # Excludes unoccupied adjacent coordinates to reduce computation
    unoccupied_adj = [coord for coord in get_adjacent_coords(start_coord) 
           if coord not in board ] 

    templates = list(_TEMPLATES.values())
    # Iterate through each piece type
    for piece_type, template in _TEMPLATES.items():
        if (sim_mode == 1) and len(templates) != 0:
            template_idx = random.randrange(0, len(templates))
            template = templates[template_idx]

        # Iterate through each original coordinate in piece_type
        for original_coord in template:
            
            # Iterate through each unoccupied adj coordinate
            for adj in unoccupied_adj:
                
                # Calculate adjusted template, create new piece
                adjusted_template = [adj + offset - original_coord for offset in template]
                new_piece = Piece(adjusted_template)
            
                # Check legality of new piece
                is_legal = True
                for coord in new_piece.coords:
                    
                    # Check cell of board is empty 
                    if coord in board:
                        is_legal = False
                        break

                    # Check coord is within board bounds
                    if not (0 <= coord.r < BOARD_N and 0 <= coord.c < BOARD_N):
                        is_legal = False
                        break

                    # Check new coord isnt overlapping starting coord
                    if start_coord == coord :
                        is_legal = False
                        break
               
                # Check the new piece is draw through one of teh adjacent coords   
                if adj not in new_piece.coords:
                    is_legal = False
                
                # If first move, dont expect any neighbors
                if turn_count == 0  :
                    if adj in new_piece.coords:
                        if is_legal:
                            valid_adjacent_pieces.append(new_piece)
                
                # If not first move and its legal, add it to valid moves
                elif is_legal:
                    valid_adjacent_pieces.append(new_piece)
        
        if (sim_mode == 1):
            if (len(valid_adjacent_pieces) != 0):
                break
            else:
                templates.pop(template_idx)

    # Remove any duplicates
    unique_valid_adjacent_pieces = set(valid_adjacent_pieces)
    
    return list(unique_valid_adjacent_pieces)




class Node:
    def __init__(self, color: PlayerColor, move: Piece):
        self.move_coord = move # store the piece of possible move
        self.player_color = color
        self.parent = None
        self.child = np.empty(0, dtype=Node)
        self.utility = 0
        self.playout = 0

    def add_parent (self, parent):
        self.parent = parent

    def add_child (self, child):
        self.child = np.append(self.child, child)

    def child_in_node(self, move: Piece, color: PlayerColor):
        child_in_current_node = False
        current_node = None
        coords_lst = [coord for coord in move.coords]
        for c in self.child:
            child_coords_lst = [coord for coord in c.move_coord.coords]
            if c.player_color == color and (coords_lst[0] in child_coords_lst) and (coords_lst[1] in child_coords_lst) and (coords_lst[2] in child_coords_lst) and (coords_lst[3] in child_coords_lst):
                child_in_current_node = True
                current_node = c
                break
        return child_in_current_node, current_node


class MCTS:
    """
    This class acts as the tree used in MCTS, which stores the root of the tree, board, and playboard for simulation
    """
    def __init__(self, color: PlayerColor, board: Dict[Coord, PlayerColor]):
        self.color = color
        self.board = board
        self.playboard = board
        self.root = None
        self.current_node = None

    
    def backpropagation (self, ending_node:Node, winner: PlayerColor):
        update_current_node = ending_node

        while update_current_node != self.current_node:
            if update_current_node.player_color == winner:
                update_current_node.utility += 1
            update_current_node.playout += 1
            update_current_node = update_current_node.parent

        if update_current_node.player_color == winner:
                update_current_node.utility += 1
        update_current_node.playout += 1   
        return

    def simulation(self, selected_node: Node, turn_count):
        self.playboard = self.board.copy()
        sim_turn_count = turn_count
        self_color = self.color
        if self_color == PlayerColor.RED:
            opp_color = PlayerColor.BLUE
        else:
            opp_color = PlayerColor.RED
        sim_current_color = self_color
        sim_current_node = selected_node

        all_possible_moves = possible_moves(self.playboard, sim_current_color, sim_turn_count, 1)

        #start_time = time.time()
        while len(all_possible_moves) != 0 and sim_turn_count < 75:
            move = random.choice(all_possible_moves)
            child_in_current_node, node = sim_current_node.child_in_node(move, sim_current_color)
            if not child_in_current_node:
                node = Node(sim_current_color, move)
                sim_current_node.add_child(node)
                node.add_parent(sim_current_node)
            
            for coord in move.coords:
                self.playboard[coord] = sim_current_color

            sim_current_node = node
            if sim_current_color == self_color:
                sim_current_color = opp_color
                sim_turn_count += 1
            else:
                sim_current_color = self_color
            
            all_possible_moves = possible_moves(self.playboard, sim_current_color, sim_turn_count, 1)
            

        if len(all_possible_moves) == 0:
            if sim_current_color == self_color:
                winner = opp_color
            else:
                winner = self_color
        else:
            no_of_self_tokens = get_colour_coordinates(self.playboard, self_color)
            no_of_opp_tokens = get_colour_coordinates(self.playboard, opp_color)
            if no_of_self_tokens > no_of_opp_tokens:
                winner = self_color
            else:
                winner = opp_color

        self.backpropagation(sim_current_node, winner)
        
        
        return 


    def selection(self, turn_count: int, no_of_sim: int) -> Node:
        all_possible_moves = possible_moves(self.board, self.color, turn_count, 0)
        for possible_move in all_possible_moves:
            child_in_current_node, node = self.current_node.child_in_node(possible_move, self.color)
            if not child_in_current_node:
                node = Node(self.color, possible_move)
                self.current_node.add_child(node)
                node.add_parent(self.current_node)
            
        ucb_max = 0 
        selected_node = self.current_node.child[0]
        childs = self.current_node.child
        for child in childs:
            exploitation_term = child.utility / child.playout if child.playout != 0 else 0
            if child.playout != 0:
                exploration_term = math.sqrt(math.log(self.current_node.playout)/child.playout) 
            elif no_of_sim > 10:
                exploration_term = 2
            else: 
                exploration_term = float("inf")
            ucb = exploitation_term + 1.41*exploration_term
            if ucb > ucb_max:
                    selected_node = child
                    ucb_max = ucb
            
        
        return selected_node

            

        

    def update_board(self, board: Dict[Coord, PlayerColor], move_player: PlayerColor, move: List[Coord]):
        self.board = board
        self.playboard = board
        
        move = Piece(move)
        if self.root == None:
            self.root = Node(PlayerColor.RED, move)
            self.current_node = self.root
            return

        child_in_current_node, node = self.current_node.child_in_node(move, move_player)

        if (node != None):
            self.current_node = node
        else:
            new_node = Node(move_player, move)
            new_node.add_parent(self.current_node)
            self.current_node.add_child(new_node)
            self.current_node = new_node

        
    def choose_move(self, turn_count):
        all_possible_moves = possible_moves(self.board, self.color, turn_count, 0)
        childs = self.current_node.child
        best_move = None
        best_score = 0

        for child in childs:
            if child.playout != 0:
                if child.move_coord in all_possible_moves:
                    if (child.utility/child.playout) > best_score:
                        if best_move == None:
                            best_move = child
                            best_score = child.utility/child.playout
                            continue
                        if child.playout >= best_move.playout:
                            best_move = child
                            best_score = child.utility/child.playout
                        elif best_move.playout > child.playout and best_score > 0.6 and (best_move.playout - child.playout) > 2:
                            continue
                        else:
                            best_move = child
                            best_score = child.utility/child.playout
                    elif (child.utility/child.playout) == best_score:
                        if best_move != None and (child.playout > best_move.playout):
                            best_move = child
        
        if (best_move == None):
            best_move = random.choice(all_possible_moves)
            return best_move

        return best_move.move_coord        



class Agent:
    """
    This class is the "entry point" for your agent, providing an interface to
    respond to various Tetress game events.
    """
    def __init__(self, color: PlayerColor, **referee: dict):
        """
        This constructor method runs when the referee instantiates the agent.
        Any setup and/or precomputation should be done here.
        """

        self.color = color
        self.board = {}
        self.turn_count = 0
        self.tree = MCTS(self.color, self.board)
        
       
        match color:
            case PlayerColor.RED:
                print("Testing: I am playing as RED")
                
            case PlayerColor.BLUE:
                print("Testing: I am playing as BLUE")


    def action(self, **referee: dict) -> Action:
        """
        This method is called by the referee each time it is the agent's turn
        to take an action. It must always return an action object. 
        """
 
        # first turn pick random
        possible_moves = []
        if self.turn_count == 0:  #1st play
            start_square = get_random_square()
            for i in get_adjacent_coords(start_square):
                possible_moves.extend(generate_valid_tetrinos(i,self.board,self.color,self.turn_count, 0))
            
            move = random.choice(possible_moves)
            
       
        #every other turn pick accoridng to MCTS
        else:
            accu_time = 0
            start_time = time.time()
            no_of_sim = 0
            while (accu_time < 5):
                selected_node = self.tree.selection(self.turn_count, no_of_sim)
                self.tree.simulation(selected_node, self.turn_count)
                no_of_sim+= 1
                end_time = time.time()
                accu_time = end_time - start_time
            move = self.tree.choose_move(self.turn_count)

        #convert Piece for PlaceAction move
        coords_lst = [coord for coord in move.coords]

        

        return PlaceAction(*coords_lst)


    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        This method is called by the referee after an agent has taken their
        turn. You should use it to update the agent's internal game state. 
        """
        # my agent update 
        if color == self.color:
            for coord in action.coords:
                self.board[coord] = self.color  # update the game board
            self.turn_count += 1 #increment accepted turn

         # not my agent update board
        else:
            for coord in action.coords:
                self.board[coord] = color  # update the game board
            
        # update board in tree        
        self.tree.update_board(self.board, color, action.coords)
        

        # if full row or column, update my board 
        remove_full_rows_and_columns(self.board)