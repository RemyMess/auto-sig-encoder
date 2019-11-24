# auto_sig_encoder
Project for the "Theory of deep learning" lecture given by Prof. Jared Taner, Oxford Math. Institute, Michaelmas 2019.

-----------------------------------
# Summary:
As nicely presented in [A Primer on the Signature Method in Machine Learning; https://arxiv.org/abs/1603.03788], the signature is a vastly rich, interpretable and simple transformation on time series. In addition to offering several computationally pleasant features (such as a feature transform allowing the linearisation of any general regression problem involving time series [Learning from the past, predicting the statistics for the future, learning an evolving system; https://arxiv.org/abs/1309.0260] among many others), the signature of a path consists in the smallest amount of information necessary to reconstruct the solution of the differential equation controlled by the former path. In particular, the signature presents itself as a natural way to compress any time series by removing all the redundancy in the information it contains.

# Goal:
In this project, we are interested to compare the theoretically well-established compressive aspect of the signature with a family of deep learning architecture that famously set itself as a standard for dimensionaly reduction, the auto-encoders (AE).

First, (write later) .....................parallel comparison

Second, (write later).................... new architecture: auto sig encoder (ASE)

Third, (write later).................... more elaborated variante: variational auto sig encoder (VASE)

# How to use the repo:
1. Run /auto-sig-encoder/build/build.sh to update the repo, install the modules and set the environment.
2. Run the experiment you want in /auto-sig-encoder/run_files.

# Technical informations and code architecture:
- Exclusively using Python 3.6.
- Repo. structured into 4 folders ("src" for the back-end computations, "data_set" for the data preparation, "build" for the building of the dependance and "run_files" to run the experiments)
- The .env environment lives in the base folder /auto-sig-encoder/; all the used modules are set in /auto-sig-encoder/build/requirements.txt 
- To build/update the environment as well as update the repo, run /auto-sig-encoder/build/build.sh
- The running of the experiment is done via the scripts in /auto-sig-encoder/run_files
