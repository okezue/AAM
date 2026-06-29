#!/usr/bin/env bash
set -euo pipefail

: "${AAM_JOB_QUEUE:?set AAM_JOB_QUEUE}"
: "${AAM_JOB_DEFINITION:?set AAM_JOB_DEFINITION}"
: "${AAM_CONFIG_S3_URI:?set AAM_CONFIG_S3_URI to a prefix containing zero-padded YAML files}"
: "${AAM_OUTPUT_S3_URI:?set AAM_OUTPUT_S3_URI}"
COUNT=${1:?usage: submit_array.sh NUMBER_OF_CONFIGS}
NAME=${AAM_JOB_NAME:-aam-confirmatory}

aws batch submit-job \
  --job-name "$NAME" \
  --job-queue "$AAM_JOB_QUEUE" \
  --job-definition "$AAM_JOB_DEFINITION" \
  --array-properties size="$COUNT" \
  --container-overrides "environment=[
    {name=AAM_CONFIG_URI,value=${AAM_CONFIG_S3_URI}/\$AWS_BATCH_JOB_ARRAY_INDEX.yaml},
    {name=AAM_OUTPUT_URI,value=${AAM_OUTPUT_S3_URI}/\$AWS_BATCH_JOB_ARRAY_INDEX/}
  ]"
