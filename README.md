prepare_data: A notebook that takes images from the original folders provided by Shailaja and prepares them for processing by the algorithm. 
The first cell construct_actin_outline reads in the files - actin, junction, outlines, and renames them with consistent numbers and adds UF or DF to indicate cell types. 
All files are written to a temp folder. The next cell allocates the files randomly to train, valid and test. The final cell prepares the leakiness files and writes them 
into a folder called permeability.

outline_leakiness_results: Shows the outcome of the models in a series of cells.

Scripts has 4 executable shells: run_outline - fits a model from actin to junctions, and a model from predicted junctions to outlines.
                                 run_junc_outline - fits a model directly from junctions to outlines.
                                 run_leakiness - fits a model from junctions to leakiness
                                 test - reads in two models actin->junction, pred_junction-> outline and runs the two consecutive models on the actin data.
