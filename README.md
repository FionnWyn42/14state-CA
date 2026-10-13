This is a cellular automta (CA) with 14 states.

The rules of this CA revolve around a lookup table which 
matches pairs of states to a new state

At each time step all cells look to 2 neighbours*
and based on those uses the lookup to find it's new state

which neighbours it looks at depends on it's current state
and the timestep, cycling through a set number of neighbourhood pairs

the code current is not very optimized so expect long run times
for anything beyond 100 x 100 grids and more than 10000 timesteps

currently you can only chose random neighbourhood configurations
based on a seed value, I would like to try and make it more precise line the 
lookup table

* currently 2 neighbours don't need to be distinct so can be only looking at 1 state
  
