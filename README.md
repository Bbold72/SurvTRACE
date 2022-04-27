# Reproducibility of SurvTRACE: Transformers for Survival Analysis with Competing Events

### Tasks

- [ ] Documentation
- [X] Citation to the original paper
- [X] Link to the original paper’s repo (if applicable)
- [X] Dependencies
- [ ] Data download instruction
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