from racetrack_helpers import *
#It already imported also numpy and all that
import pytest

def test_loading():
    draha = load_racetrack("draha_01.png")
    #racetrack must a np array
    assert isinstance(draha, np.ndarray)
    #2 dimensions and shape (40, 40)
    assert draha.shape == (40, 40)
    #draha needs a start, finish, wall and racetrack itself
    assert 0 in draha
    assert 1 in draha
    assert 2 in draha
    assert 3 in draha


def test_list_8_tiles_sourrounding_the_middle_one():
    our_list = list_8_tiles_sourrounding_the_middle_one((4,5))
    #We need 8 tiles
    assert len(our_list) == 8
    #We need all 8 tiles to be unique
    unique_list = list(set([tuple(i) for i in our_list]))
    assert len(unique_list) == 8

def test_astar():
    draha = load_racetrack("draha_01.png")
    list_starts = np.asarray(np.where(draha == 2)).T.tolist()
    list_finishes = np.asarray(np.where(draha == 3)).T.tolist()
    start = random.choice(list_starts)
    finish = random.choice(list_finishes)
    _, astar_path, astar_distances = astar(draha, start = start, 
                                           finish = finish, symbol_wall = 1)
    #on this map path obviously exists so the algorithm should find it:
    assert isinstance(astar_path, list)
    assert len(astar_path) > 0
    assert len(astar_path) == len(astar_distances)
    # start and finish both should be in the path found
    assert start in astar_path
    assert finish in astar_path
    
def test_create_evaluation_matrix_v01():
    draha = load_racetrack("draha_01.png")
    list_starts = np.asarray(np.where(draha == 2)).T.tolist()
    list_finishes = np.asarray(np.where(draha == 3)).T.tolist()
    start = random.choice(list_starts)
    finish = random.choice(list_finishes)
    _, astar_path, astar_distances = astar(draha, start = start, 
                                           finish = finish, symbol_wall = 1)
    eval_matrix, wall_value = create_evaluation_matrix_v01(draha, astar_path, astar_distances, list_finishes)
    #eval matrix must be a np.ndarray and must contain certain values
    assert isinstance(eval_matrix, np.ndarray)
    assert wall_value in eval_matrix
    assert 0 in eval_matrix
    assert eval_matrix.shape == draha.shape
    #wall value must be an integer (or float)
    assert isinstance(wall_value, int) or isinstance(wall_value, float)
    
def test_tiles_to_move_to():
    current_position = (5,6)
    vector = (3,5)
    max_speed = 10
    eval_matrix = np.zeros(shape=(64,64)) #We use it only to get the shape, so I don't actually need to load it
    tiles = tiles_to_move_to(current_position, vector, max_speed, eval_matrix)
    #Let's check the basics first
    assert isinstance(tiles, np.ndarray)
    assert tiles.shape[0] == 8
    assert tiles.shape[1] == 2
    #What if the vector is too big?
    tiles_2 = tiles_to_move_to(current_position, (5,11), max_speed, eval_matrix)
    print(tiles_2)
    assert tiles_2.shape[0] == 0
    #What if we wanna move to tiles that are out of bounds?
    tiles_3 = tiles_to_move_to((1,1), (-4,-4), max_speed, eval_matrix)
    assert tiles_3.shape[0] == 0
    

def test_formula_greedy_v03():
    draha = load_racetrack("draha_01.png")
    list_starts = np.asarray(np.where(draha == 2)).T.tolist()
    list_finishes = np.asarray(np.where(draha == 3)).T.tolist()
    start = random.choice(list_starts)
    _, astar_path, astar_distances = astar(draha, start = start, finish = random.choice(list_finishes), symbol_wall = 1)
    eval_matrix, wall_value = create_evaluation_matrix_v01(draha, astar_path, astar_distances, list_finishes)
                
    fm_path_AI = formula_greedy_v03(start, eval_matrix, wall_value=10**9, max_speed=10)
    #Check if the path
    assert isinstance(fm_path_AI, list)
    assert start in fm_path_AI
    #Check if one of the tiles of finish is in the path
    finish_in_path = False
    for (x,y) in fm_path_AI:
        if eval_matrix[x,y] == 0:
            finish_in_path = True
            break
    assert finish_in_path == True
    #Check if any wall tile is in the path
    wall_in_path = False
    for (x,y) in fm_path_AI:
        if eval_matrix[x,y] == 10**9:
            wall_in_path = True
            break
    assert wall_in_path == False
    
def test_move_current_tile():
    current_tile = (5,6)
    possible_tiles = [[5,6], [5,7], [5,4], [4,4], [3,4], [3,5], [3,6], [4,6]]
    new_tile = move_current_tile(current_tile, possible_tiles, key = "left")
    assert new_tile == [5,4]
    new_tile_2 = move_current_tile(current_tile, possible_tiles, key = "right")
    assert new_tile_2 == [5,7]
    new_tile_3 = move_current_tile(current_tile, possible_tiles, key = "up")
    assert new_tile_3 == [4,6]
    new_tile_4 = move_current_tile(current_tile, possible_tiles, key = "down")
    assert list(new_tile_4) == [5,6] #it the type it recieved if the tile did not change
    
#The only two things I am not testing is display_rt() because it returns a matplotlib object
#and that is a weird type to work with (and it works well normally)
#and give_formula_path_v01() because it is basically just a for cycle inside which I call formula_greedy_v03()
