\ingroup GrpDummyPages
\page OpenPARF_Database Database

We introduce class Layout to represent the FPGA architecture and class Design to represent the circuit netlist.
In a design object, the key is to store the circuit netlist (a hypergraph essentially), which is represented by three arrays: instance array, net array, and pin array.
The relationships between the three arrays are also plotted in the figure.
An instance may contain multiple pins, each of which incident to a net (suppose we ignore floating pins).
In real implementation, we introduce class Model like a standard cell in ASIC, wrapping a netlist for hierarchical designs.
An instance is an instantiation of a model.
The top design is a special class containing all information from class Model.

![Database Representation](@ref db_repr.svg "db_repr.svg")
@image latex db_repr.pdf "Database Representation" width=\textwidth

Although we design the netlist data structure to consider hierarchical representation, we have not incorporated the functionality for flattening a netlist,
which is usually the required input for placement and routing algorithms.
This is left to the future work.

In the layout part, to make representation general to a wide range of FPGA architectures, we introduce the mapping between **Model**, **Resource**, and **Site Type**.
The most challenging part is to handle LUT related resources properly.
Other resources can only be assigned to a specific type of sites.
In this general representation, we model the mapping like a graph, with a model consuming one or multiple kinds of resources, and a site containing one or multiple kinds of resources.
The mapping automatically generates indices for different types, and will not rely on hard-coded naming for indexing.

However, despite this general representation, when it comes to placement and routing algorithms, we do need to know exactly LUTL, LUTM, FF, Carry, etc.
That is when we generate **PlaceDB** and **RouteDB** (Note that algorithms will only interact with PlaceDB or RouteDB).
Therefore, we have to sacrifice some degrees of generality and introduce an *enum* type **ResourceCategory**.
It contains hard-coded entries like *LUTL*, *LUTM*, *FF*, and *Carry*.
It also has generic entries like *SSSIR* (single-site single-instance resource) and *SSMIR* (single-site multiple-instance resource) for DSP, RAM, IO, etc.
Note that each resource only corresponds to one resource category, but each model may point to multiple resource categories, as a model can consume multiple resources.
All these complicated schemes are introduced to handle LUTL and LUTM.
