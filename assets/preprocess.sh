#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ggen

while [[ $# -gt 0 ]]
do
  key="${1}"

  case ${key} in
  -t|--target)
    target="${2}"
    shift 2
    ;;
  -h|--help)
    echo "Description"
    shift # past argument
    ;;
  *)    # unknown option
    shift # past argument
    ;;
  esac
done

if [ -z "$target" ]; then
  echo 'Missing -t or --target' >&2
  echo 'example: bash preprocessing.sh --target exampleA' >&2
  exit 1
fi

python mk_edge.py --target $target
python mk_noise.py --target $target
