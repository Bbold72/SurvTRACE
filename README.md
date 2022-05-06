# Reproducibility of SurvTRACE: Transformers for Survival Analysis with Competing Events

### Tasks

- [ ] Documentation
- [X] Citation to the original paper
- [X] Link to the original paper’s repo (if applicable)
- [X] Dependencies
- [X] Data download instruction
- [ ] Preprocessing code + command (if applicable)
- [ ] Training code + command (if applicable)
- [ ] Evaluation code + command (if applicable)
- [ ] Pretrained model (if applicable)
- [ ] Table of results (no need to include additional experiments, but main reproducibility result should be included)

### Original Paper and Repository

This repository is based and inspired on the following work:

```none
Zifeng Wang and Jimeng Sun. 2021. SurvTRACE: Transformers for Survival Analysis with Competing Events.
```

You can find the paper at [https://arxiv.org/abs/2110.00855](https://arxiv.org/abs/2110.00855) and the repository at [https://github.com/RyanWangZf/SurvTRACE](https://github.com/RyanWangZf/SurvTRACE).

### How to configure the environment

Use our pre-saved conda environment!

```bash
conda env create --name survtrace --file=survtrace.yml
conda activate survtrace
```

or try to install from the requirement.txt

```bash
pip3 install -r requirements.txt
```

### How to get the data

For this project we use different datasets to run our experiments.

* Study to Understand Prognoses Preferences Outcomes and Risks of Treatment (SUPPORT) (Knaus et al. 1995).
* Molecular Taxonomy of Breast Cancer International Consortium (METABRIC) (Curtis et al. 2012).
* Surveillance, Epidemiology, and End Results Program (SEER).

[`pycox`](https://github.com/havakv/pycox) provides the SUPPORT and METABRIC [datasets](https://github.com/havakv/pycox#real-datasets). Meanwhile, access to SEER has to be requested as the instructions in the following sub section.

#### How to get the SEER dataset.

1. Go to https://seer.cancer.gov/data/ to ask for data request from SEER following the guide there.

2. After complete the step one, we should have the **seerstat software** for data access. Open it and sign in with the username and password sent by seer.

3. Use seerstat to open the **./data/external/seer.sl** file. Click on the 'excute' icon to request from the seer database. We will obtain a csv file.

4. Move the csv file to **./data/raw/seer_raw.csv**, then run script to create the processed data, as

   ```shell
   make seer
   ```

   we will obtain the processed seer data named **seer_processed.csv** located in **./data/processed/**.

### Running the experiments

You can run all the steps for creating the results with the following command.

```shell
make run
```

Alternatively you could do each step separately:

1. Clean previous generated files, if they exist.

   ```shell
   make clean
   ```

2. Process SEER dataset.

   ```shell
   make seer
   ```

3. Generate datasets. By default it does 10 runs, but you can change the NUM_RUNS argument.

   ```shell
   make datasets [-e NUM_RUNS=10]
   ```

