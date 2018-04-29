from gym.envs.registration import register

register(
    id='Maze-v0',
    entry_point='Maze.envs.Maze:maze'
)

register(
    id='Maze-v1',
    entry_point='Maze.envs.maze1:maze'
)

register(
    id='Maze-v2',
    entry_point='Maze.envs.maze2:maze'
)
