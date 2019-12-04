# Data mining (CS-E4600): graph partitioning

## About
Assignment for data mining course.

## Step to use the code
1. Install all the python module requirements listed in requirements.txt
2. ``cd code``
3. Run main.py
    * ``python main.py --graphName <graphName>``
        * To run a grid search optimization <graphName>.txt
    * ``python main.py --graphName all``
        * To run a grid search optimization on the 5 graphs of the statement
        

## Competition
We didn't take part in the competition

## Data

These graphs are not necessarily connected and they may contain a large number of connected components, which might result in trivial solutions (in particular, when the value of k is less than the number of connected components). Henceforth, if a graph is not connected, by convention,
we will work with its largest connected component.

* [graphs from the Stanford Network Analysis Project (SNAP)](http://snap.stanford.edu/data/index.html)


## Authors

Antoine Louis (784915) & Olivier Moitroux (784928)


