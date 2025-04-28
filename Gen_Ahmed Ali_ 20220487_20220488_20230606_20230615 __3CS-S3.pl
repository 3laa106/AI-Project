%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Problem#1%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%--------- MAIN SOLVE ---------
solve :-
    example_grid(Grid),
    % Show initial grid
    write('Initial Grid:'), nl,
    draw_grid(Grid, [], (none, none)), nl,

    % Find and show steps
    find_best_path(Grid, BestPath, MaxDeliveries),
    write('Steps:'), nl,
    draw_steps(Grid, BestPath), nl,

    % Show final grid separately
    write('Final Grid:'), nl,
    last(BestPath, LastPos),
    draw_grid(Grid, BestPath, LastPos), nl,

    % Show total deliveries
    format('Total Deliveries: ~w~n', [MaxDeliveries]).

% --------- FIND BEST PATH ---------
find_best_path(Grid, BestPath, MaxDeliveries) :-
    length(Grid, Rows),
    ( Rows > 0 -> nth0(0, Grid, FirstRow), length(FirstRow, Cols) ; Cols = 0 ),
    find_drone_start(Grid, StartX, StartY),
    count_ps(Grid, TotalPs),  % Count total 'P's in the grid
    empty_visited(Visited),
    empty_path(Path),
    findall(
        [Deliveries, FullPath],
        dfs(Grid, Rows, Cols, StartX, StartY, Visited, Path, 0, TotalPs, Deliveries, FullPath),
        Paths
    ),
    ( Paths = [] ->
        BestPath = [], MaxDeliveries = 0
    ;
        find_max_deliveries(Paths, BestPath, MaxDeliveries)
    ).

% Count all 'P's in the grid
count_ps(Grid, Count) :-
    flatten(Grid, FlatGrid),
    include(==('P'), FlatGrid, Ps),
    length(Ps, Count).

% --------- FIND DRONE START POSITION ---------
find_drone_start(Grid, X, Y) :-
    nth0(Y, Grid, Row),
    nth0(X, Row, 'D'),
    !.

% --------- DEPTH FIRST SEARCH (DFS) ---------
dfs(Grid, Rows, Cols, X, Y, Visited, Path, CurrentDeliveries, TotalPs, TotalDeliveries, FullPath) :-
    (CurrentDeliveries =:= TotalPs ->  % Stop if all 'P's visited
        TotalDeliveries = CurrentDeliveries,
        FullPath = Path
    ;
        valid_position(Rows, Cols, X, Y),
        nth0(Y, Grid, Row),
        nth0(X, Row, Cell),
        Cell \= 'O',
        \+ memberchk((X,Y), Path),
        append(Path, [(X,Y)], NewPath),
        ( Cell = 'P', \+ visited(Visited, (X,Y)) ->
            NewDeliveries is CurrentDeliveries + 1,
            mark_visited(Visited, (X,Y), NewVisited)
        ;
            NewDeliveries = CurrentDeliveries,
            NewVisited = Visited
        ),
        ( member([Dx, Dy], [[1,0], [0,1], [-1,0], [0,-1]]),
          NewX is X + Dx,
          NewY is Y + Dy,
          dfs(Grid, Rows, Cols, NewX, NewY, NewVisited, NewPath, NewDeliveries, TotalPs, TotalDeliveries, FullPath)
        ;
          TotalDeliveries = NewDeliveries,
          FullPath = NewPath
        )
    ).

% --------- VALID POSITION ---------
valid_position(Rows, Cols, X, Y) :-
    X >= 0, X < Cols,
    Y >= 0, Y < Rows.

% --------- VISITED HELPERS ---------
empty_visited([]).
visited(Visited, Pos) :- memberchk(Pos, Visited).
mark_visited(Visited, Pos, [Pos|Visited]).

empty_path([]).

% --------- FIND MAXIMUM DELIVERIES ---------
find_max_deliveries([First|Rest], BestPath, MaxDeliveries) :-
    First = [FirstDeliveries, FirstPath],
    find_max_deliveries(Rest, FirstDeliveries, FirstPath, BestPath, MaxDeliveries).

find_max_deliveries([], CurrentMax, CurrentPath, CurrentPath, CurrentMax).
find_max_deliveries([[Deliveries, Path]|Rest], CurrentMax, CurrentPath, BestPath, MaxDeliveries) :-
    ( Deliveries > CurrentMax ->
        find_max_deliveries(Rest, Deliveries, Path, BestPath, MaxDeliveries)
    ;
        find_max_deliveries(Rest, CurrentMax, CurrentPath, BestPath, MaxDeliveries)
    ).

% --------- DRAW GRID ---------
draw_grid(Grid, Path, (CurrX, CurrY)) :-
    draw_grid_rows(Grid, 0, Path, (CurrX, CurrY)).

draw_grid_rows([], _, _, _).
draw_grid_rows([Row|Rest], Y, Path, (CurrX, CurrY)) :-
    draw_grid_row(Row, Y, 0, Path, (CurrX, CurrY)), nl,
    Y1 is Y + 1,
    draw_grid_rows(Rest, Y1, Path, (CurrX, CurrY)).

draw_grid_row([], _, _, _, _).
draw_grid_row([Cell|Rest], Y, X, Path, (CurrX, CurrY)) :-
    ( (X, Y) = (CurrX, CurrY) ->
        write('D')
    ; member((X,Y), Path) ->
        write('*')
    ; write(Cell)
    ),
    write(' '),
    X1 is X + 1,
    draw_grid_row(Rest, Y, X1, Path, (CurrX, CurrY)).

% --------- DRAW STEPS ONE BY ONE ---------
draw_steps(_, []).
draw_steps(Grid, [Current|Rest]) :-
    draw_steps(Grid, Rest, [Current]).

draw_steps(_, [], _).
draw_steps(Grid, [Current|Rest], Visited) :-
    append(Visited, [Current], NewVisited),
    draw_grid(Grid, NewVisited, Current), nl,
    draw_steps(Grid, Rest, NewVisited).

% --------- PRINT FINAL ROUTE ---------
print_route(Grid, Path) :-
    draw_route_rows(Grid, Path, 0).

draw_route_rows([], _, _).
draw_route_rows([Row|Rest], Path, Y) :-
    draw_route_row(Row, Path, Y, 0), nl,
    Y1 is Y + 1,
    draw_route_rows(Rest, Path, Y1).

draw_route_row([], _, _, _).
draw_route_row([Cell|Rest], Path, Y, X) :-
    ( member((X,Y), Path) ->
        ( Cell = 'O' -> write('O') ; write('-') )
    ; write(Cell)
    ),
    write(' '),
    X1 is X + 1,
    draw_route_row(Rest, Path, Y, X1).

% --------- EXAMPLE GRID ---------


example_grid([
    ['D', '-', 'P', '-', 'O'],
    ['-', 'O', '-', '-', 'P'],
    ['-', '-', 'O', 'P', '-'],
    ['P', 'O', '-', '-', '-'],
    ['-', '-', 'P', 'O', '-']
]).
/*
example_grid([
    ['D', '-', 'P'],
    ['-', 'O', '-'],
    ['-', '-', 'P']
]).
*/


/*
example_grid([
    ['D', '-', 'P', '-', 'O', '-'],
    ['-', 'O', '-', '-', 'P', 'O'],
    ['-', '-', 'O', 'P', '-', '-'],
    ['P', 'O', '-', '-', '-', '-'],
    ['-', '-', 'P', 'O', '-', 'O'],
    ['-', '-', 'P', '-', 'O', '-']
]).
*/
/*
example_grid([
    ['D', '-', 'P', '-', 'O', '-', '-', 'P'],
    ['-', 'O', '-', '-', 'P', 'O', 'O', '-'],
    ['-', '-', 'O', 'P', '-', '-', 'O', '-'],
    ['P', 'O', '-', '-', '-', 'P', 'O', '-'],
    ['-', '-', 'P', 'O', '-', '-', 'P', '-'],
    ['-', '-', 'P', '-', 'O', '-', 'P', '-'],
    ['-', '-', 'O', '-', '-', '-', '-', '-'],
    ['-', 'P', '-', 'O', 'P', 'O', '-', 'P']
]).
*/

