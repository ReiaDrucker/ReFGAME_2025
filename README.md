# ReFGAME_2025
Code base used in Representative Features for Game Agnostic Movement Evaluation (ReFGAME)

# Reproducibility Info
In order to run this code it is recommended that you create a virtual environment with Python 3.11.7 and install the packages listed in requirements.txt.
We used PyCharm 2023.1.1.1 as an IDE when writing this code so try it if you run into any issues.

Included in this repo are four parquet files under /Outputs that have the raw computed features for player movement and camera movement in rounds and in halves. 
The main results and figures from the paper are generated using Logistic_Regression_Models.ipynb, Journey_Dwell_Distributions_3D_Halves.ipynb, and csgo_players_halves.ipynb with some other notebooks included as examples/reference to other work explored.
Secondary to this are some examples on how to use awpy 1.3.1 for CSGO analysis pulled from an older version of https://github.com/pnxenopoulos/awpy as they have now moved on to version 2 to support CS2 and the resources for CSGO can be somewhat hard to find now. They are included here under "/awpy examples" to aid in any future work but are NOT our work.

If you wish to run all the code here from scratch and generate any intermediate files you will need to first download the ESTA dataset from https://github.com/pnxenopoulos/esta and copy the compressed demo files into the demo-files folder.
The full structure of which should be demo-files/esta-main/data/lan.

Then run the following files in order (you can use pipeline.sh to make this easier):
    "Parsing/Parse_ESTA_LAN.py"
    "Compute_Features/Journey_Dwell_Compute.py"
    "Compute_Features/Journey_Dwell_Metrics_Compute.py"
    "Compute_Features/Refgem_Compute.py"
    "Compute_Features/Collate_Features.py"

This will take a LONG TIME and is not recommended for anyone who wants to just explore the data. 
If you are interested in our implementation for feature computation see the /Compute_Features folder and the files Refgem_Compute.py and Collate_Features.py in particular.


