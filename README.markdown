# Minesweeper Solver Prototype
This is a casual project that I'm currently working on to gain experience with C++. The goal of this project is to create an algorithm to solve Minesweeper on expert difficulty 45% of the time as fast as possible. Currently on my Macbook Pro with an Apple M1 chip, it can solve 42.66% of expert games with seeds 0-9999 in 55 seconds making the average game from this set solvable in 5.5 millisecs.

I'm in the process of rewriting this project, the rewrite code can be found on the rewrite branch of this project.
## Planned Fixes and Improvements in rewrite:
- Add in the ability for the user to play minesweeper.
- Optimize the algorithm further by using the techinique specified [here](https://minesweepergame.com/strategy/patterns.php) which will reduce the amount of calls to find all solutions (An NP time complexity function).
- Implement memory pooling to reduce heap allocations.
- Factor in reguess probaility when geussing to reduce guesses.
- Implement multithreading so program can take inputs while its running algorithm.
- Swap terminal UI with a GUI to make it more user friendly.
- Improve readability of code by choosing better names for variables and functions.
## How it works
- A class called "mineMap" emulates the actual Minesweeper game.
- A class called "solver" receives the game state through a string.
- The solver class will then search for any tiles that have the same number of adjacent bombs as adjacent unknown tiles. If this is true, all adjacent unknown tiles will be flagged.
- The solver class will then search for any tiles that have the same number of adjacent bombs as adjacent flagged tiles. If this is true, all adjacent unknown tiles will be clicked.
- If neither of these are true, the solver class will group the tiles together into distinct groups and send these to the "probabilityFinder" class and check for all possible arrangements of bombs within these groups.
- If there is a possibility that the bombs in these groups can surpass the amount of bombs left, the algorithm will combine these solutions and exclude combinations that surpass the amount of bombs left.
- Using these solutions, the algorithm will calculate the probability of each tile being a bomb. Any tiles that have a 100% chance of being a bomb are flagged and any that have a 0% chance of being a bomb are clicked.
- If there are no obvious tile choices, the algorithm will find the bomb with the lowest probability of being a bomb and click it. If there's a tie between probabilities, the algorithm will use whether or not the tile is a corner or edge as a tie breaker.
- This cycle will be repeated every iteration.
## How to compile and run
In order to compile the program run:
```bash
cd /path/to/directory/Mine-Sweeper-Solver
mkdir build
cd build
cmake ..
make
```
You can then run the program by running
```bash
# On mac / linux:
./MineSweeperSolver
# On windows:
./MineSweeperSolver.exe
```
## How to use
When booting up the program in terminal you will be greeted by 3 options:
```
Type "w" to watch minesweeper algorithm
Type "t" to test minesweeper algorithm
Type "q" to quit
```
- Typing w will allow you to watch the minesweeper algorithm solve a map with the selected difficulty and seed.
- Typing t will run the minesweeper algorithm on seeds 0-(run_amount-1) on the selected difficulty and tell you the time taken and win rate.
- Typing q will exit the program.
## Settings
Settings can be set in the settings.txt file found in the doc directory.
- run_amount specifies how many times the algorithm is run when testing preformance and win rate. 
- difficulty has three different settings:
    - Beginner: 9x9 board, 10 bombs
    - Intermediate: 16x16 board, 40 bombs
    - Expert  30x16 board, 99 bombs
- difficulty is used when testing and when running a seed to watch.
- seed is a setting used for watching the algorithm, it is not used for testing. If you set it to "r" it will set the seed equal to time, setting it to a number will make the seed that number.
- wait_time is the sleep time in microseconds between iterations which is used when watching the algorithm not for testing.

## Terminal Grabs:
Info about coordinates and what "c x y" means:
- c x y: This refers to the coordinates of where the last click was.
- x = 0 refers the left of the screen, y = 0 refers to the top of the screen.
```
ASCII key for how minesweeper game is displayed:
- '@': Flagged
- '#': Unknown
- ' ': No adjacent bombs
- 'n' where n is an integer between 1 and 8: the adjacent bomb count
- 'X': A detonated bomb
```

Example of solved game:
```
c 29 15
iteration: 292
seed 1706645208
guesses: 0
Flags: 0
    1 @ @ 2 1 1 1 1   1 @ 2 1 1   1 1 2 1 2 2 @ @ 1 1 1 1   
    1 2 3 @ 1 1 @ 2 1 1 1 2 @ 3 2 2 @ 3 @ 3 @ 4 3 2 1 @ 3 2 
        2 2 2 2 3 @ 1     1 2 @ @ 2 1 3 @ 3 1 2 @ 2 2 2 @ @ 
1 1 1 1 2 @ 1 1 @ 2 1     1 3 4 3 1   1 1 2 2 3 3 @ 1 1 2 2 
2 @ 1 1 @ 2 1 1 2 2 2 1 1 1 @ @ 2 1 1 1 1 1 @ @ 4 2 2 1 1   
@ 2 2 2 3 3 2 1 1 @ 2 @ 1 1 2 3 @ 1 1 @ 2 2 4 @ @ 1 1 @ 1   
1 1 2 @ 3 @ @ 1 1 1 2 2 2 1 1 2 2 1 2 2 3 @ 2 2 2 1 2 2 3 1 
    2 @ 4 3 3 2 1 2 1 2 @ 1 1 @ 3 3 4 @ 4 2 2       1 @ 2 @ 
1 2 3 2 3 @ 2 1 @ 3 @ 3 1 1 1 2 @ @ @ @ 3 @ 1 1 2 2 2 1 3 2 
1 @ @ 2 2 @ 2 1 1 3 @ 3 1   1 3 5 5 4 2 2 1 1 1 @ @ 2 1 2 @ 
1 3 @ 2 1 1 1   1 2 3 @ 1 1 2 @ @ @ 2 1       1 2 2 2 @ 3 2 
  1 1 1     1 2 3 @ 2 1 1 2 @ 5 4 3 @ 1     1 1 1 1 2 4 @ 2 
  1 1 1   1 2 @ @ 3 2 1 1 2 @ @ 1 1 1 1 1 1 2 @ 1 1 @ 3 @ 2 
  1 @ 1   1 @ 4 @ 3 2 @ 2 2 3 3 2 1   1 2 @ 2 2 3 4 4 4 2 1 
1 2 3 2 1 1 2 3 3 @ 2 1 2 @ 1 1 @ 1   1 @ 3 3 3 @ @ @ @ 2 1 
1 @ 2 @ 1   1 @ 2 1 1   1 1 1 1 1 1   1 1 2 @ @ 4 @ @ 4 @ 1 
Solved!
```
Example of lost game:
```
iteration: 194
seed 1706646306
guesses: 3
Flags: 2
    1 @ 1         1 1 1 1 1 2 1 1   1 @ 2 1 1 2 @ 2 1 @ 1   
    1 2 2 1       2 @ 2 1 @ 3 @ 4 2 2 1 3 @ 2 2 @ 2 1 1 1   
1 1   1 @ 2 1     2 @ 3 2 1 3 @ @ @ 2   2 @ 2 2 2 2         
@ 3 2 2 3 @ 1     1 2 @ 3 2 2 3 @ @ 2   1 2 2 2 @ 1   1 1 1 
@ @ 2 @ 2 1 1     1 2 3 @ @ 2 3 4 3 1     1 @ 2 1 1 1 2 @ 1 
@ 3 3 2 2     1 1 2 @ 2 2 3 3 @ @ 1 1 1 1 1 1 1     1 @ 4 3 
2 2 3 @ 2     1 @ 2 1 1   1 @ 3 3 3 3 @ 2 1         1 2 @ @ 
1 @ 4 @ 2     1 1 1       1 1 1 1 @ @ 3 @ 2 2 2 2 2 1 2 3 X 
3 4 @ 3 3 2 1         1 1 1   1 2 3 2 2 2 @ 2 @ @ 3 @ 2 3 # 
@ @ 3 2 @ @ 2 1 2 2 2 2 @ 1 1 3 @ 3 1 1 2 2 3 3 @ 3 1 2 @ @ 
3 @ 3 2 3 @ 3 2 @ @ 2 @ 3 2 1 @ @ 4 @ 1 2 @ 4 3 2 1 1 2 3 2 
2 4 @ 2 1 2 @ 2 2 2 2 2 @ 1 1 3 @ 3 2 2 4 @ @ @ 3 2 4 @ 2   
# # @ 4 1 2 1 1       1 1 1   1 1 1 1 @ 3 @ @ @ 4 @ @ @ 3 1 
1 3 @ 3 @ 2 1 1                 1 1 2 1 2 2 3 3 4 @ 6 4 3 @ 
1 2 1 2 1 3 @ 3 1 1 2 2 1       1 @ 1         1 @ 3 @ @ 2 1 
@ 1       2 @ @ 1 1 @ @ 1       1 1 1         1 1 2 2 2 1   
Explode at 29 7
```
Example of test results on expert set to 10000 runs:
```
######################################################
Total Games: 10000

Wins:        4193
Loses:       5807

Win Rate:    41.93%
######################################################
Time Taken:         42.7533 secs
Time Taken Per Run: 4.27533 millisecs per run

######################################################
```
## License
Minesweeper Solver is liscenced under the MIT License, see [LICENCE.txt](https://github.com/garthable/Mine-Sweeper-Solver/blob/main/LICENSE.txt) for more information
