# Compute launch templates

The experiment code is scheduler-agnostic. These templates intentionally contain no account IDs,
credentials, AMI IDs, subnets, security groups, or storage buckets.

## Slurm

1. Materialize one YAML per condition:

   ```bash
   PYTHONPATH=src python scripts/materializeablationmatrix.py \
     configs/ablations/minimal_confirmatory.yaml \
     --outputdir runs/materialized_configs --max-runs 100
   ```

2. Create a sorted manifest:

   ```bash
   find runs/materialized_configs -name '*.yaml' -print | sort > runs/config_manifest.txt
   ```

3. Submit an array after editing partition/account/module settings:

   ```bash
   sbatch --array=0-$(($(wc -l < runs/config_manifest.txt)-1)) \
     infrastructure/slurm/aam_array.sbatch runs/config_manifest.txt
   ```

## AWS Batch

`aws_batch/job_definition.template.json` and `submit_array.sh` define an environment-variable-only
boundary. Build and push the repository Docker image, register the job definition after substituting
its placeholders, then provide `AAM_JOB_QUEUE`, `AAM_JOB_DEFINITION`, `AAM_CONFIG_S3_URI`, and
`AAM_OUTPUT_S3_URI`. The submit script does not create IAM, networking, buckets, or credentials.

Secrets must be supplied through the scheduler's secret mechanism, never embedded in YAML or run
artifacts. Closed-provider API jobs should use separate queues and least-privilege secret access.
