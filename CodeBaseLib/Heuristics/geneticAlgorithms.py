"""
Description
    -> Simulation of sexual reproduction and mutations

Evolutionary Algorithm
1. Determine a way to code the building blocks

Example
->  Optimize the design parameters for an electrical circuit.
1.  Define each component, parameters, characterizations.
2.  Enumerate the parameters in a list
3.  A particular order or selection of parameters is called "The genetic code"
4.  Generate randomly thousands of genetic codes, Each of code is called an "individual" or "solution" or "organism"
5.  Now we evaluate the "organism" in a simulated environment by using a defined method to assess each set of parameters
6.  The Organisms are sampled with an estimated Probability from the evaluation step.
7.  COMPETITION.
    Those who are selected (with reposition) are "survivors" and those who didn't are eliminated.
    Until the original population size recovers
8.  OPTIONAL, sexual reproduction. 
     The Organisms are paired and their code gets recombined
9.  Mutation. 
        We allow some random change in each organism.
10.  A Generation has Occur. 
11. Measure how much the designs have improved
12. Repeat for many generations.


"""