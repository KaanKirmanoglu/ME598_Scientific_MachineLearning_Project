#!/bin/bash
kTs=($(seq 0.1 0.1 3.0))
rho=0.3
run_nos=(22 23 24)
version=2
run_steps=1000000
dump_period=10

for run_no in ${run_nos[@]};
do
  for kT in ${kTs[@]};
  do
    log_file_name="log_version$version+_rho$rho+_kT$kT+_runid$run_no.log"
    python3 -W ignore diffusionv02.hoomd --kT $kT --rho $rho --run_no $run_no --version $version --run_steps $run_steps --dump_period $dump_period > "$log_file_name"
  done
done
