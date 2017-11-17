#!/bin/bash
# bash script to set up a suite of Thwaites variability runs


# ===========================

# Manually set/check these variables before running!

SETUP=1
ENABLE_RESTART=0
SUBMIT_RUN=0

BASE_DIR=`pwd`   # Path in which all the versions will be set up
TEMPLATE_DIR=/scratch1/scratchdirs/hoffman2/thwaites_variability/1-8km_smoketest_template

amplitudes='150 300'
periods='02 20 70'
phases="0.0 1.0 2.0"  # Note this has to be customized for each period, b/c units are years

#  ==========================



nlfile="namelist.landice"

if [ $SETUP = 1 ]; then
   # First make a 'blank' template copy of the directory
   cp -r $TEMPLATE_DIR $BASE_DIR/run_template
   cd $BASE_DIR/run_template
   # remove some unneeded garbage
   rm ./make_graph_file.py ./metis ./setup_model.py
   # remove symlink to a few files
   cp --remove-destination `readlink albany_input.xml` albany_input.xml
   cp --remove-destination `readlink slurm.wolf.run` slurm.wolf.run
   cp --remove-destination `readlink slurm.edison_bundle.run` slurm.edison_bundle.run
fi

# now create all the instances in a flat dir structure
cd $BASE_DIR
for amp in $amplitudes; do
   for per in $periods; do

      # create new edison submit script for this amp/per combo
      if [ $SETUP = 1 ]; then
         cd $BASE_DIR
         cp run_template/slurm.edison_bundle.run slurm.edison_bundle.amp${amp}_per${per}.run
         sed -i.SEDBACKUP "s/^#SBATCH -J.*/#SBATCH -J amp${amp}_per${per}_bundle/" slurm.edison_bundle.amp${amp}_per${per}.run
         rm slurm.edison_bundle.amp${amp}_per${per}.run.SEDBACKUP
      fi

      for pha in $phases; do
          # build the dir name
          dirname=amp${amp}_per${per}_pha${pha}

          # === Setting up the run ===
          if [ $SETUP = 1 ]; then
             echo Setting up:  $dirname
             cp -r $BASE_DIR/run_template $BASE_DIR/$dirname

             # update the nl settings
             cd $BASE_DIR/$dirname
             sed -i.SEDBACKUP "s/config_basal_mass_bal_seroussi_amplitude.*/config_basal_mass_bal_seroussi_amplitude = $amp/" $nlfile
             sed -i.SEDBACKUP "s/config_basal_mass_bal_seroussi_period.*/config_basal_mass_bal_seroussi_period = $per/" $nlfile
             sed -i.SEDBACKUP "s/config_basal_mass_bal_seroussi_phase.*/config_basal_mass_bal_seroussi_phase = $pha/" $nlfile
             rm $nlfile.SEDBACKUP

             # update the job name in the wolf run
             sed -i.SEDBACKUP "s/^#SBATCH --job-name.*/#SBATCH --job-name=$dirname/" slurm.wolf.run
             rm slurm.wolf.run.SEDBACKUP

             # add this run to the edison bundle
             echo "cd ${BASE_DIR}/${dirname}" >> ../slurm.edison_bundle.amp${amp}_per${per}.run
             echo  "srun -n 240 -N 5 --cpu_bind=cores /global/project/projectdirs/piscees/mpas-hoffman/MPAS-for-thwaites/landice_model &" >> ../slurm.edison_bundle.amp${amp}_per${per}.run
          fi

          # === Enabling restart ===
          if [ $ENABLE_RESTART = 1 ]; then
             cd ${BASE_DIR}/${dirname}  # redundant if also setting up, but needed otherwise
             sed -i.SEDBACKUP "s/config_do_restart.*/config_do_restart = .true./" $nlfile
             sed -i.SEDBACKUP "s/config_do_restart.*/config_start_time = 'file'/" $nlfile
             rm $nlfile.SEDBACKUP
          fi


          # === Submitting the run ===
          if [ $SUBMIT_RUN = 1 ]; then
             cd ${BASE_DIR}/${dirname}  # redundant if also setting up, but needed otherwise

             echo submitting job for run $dirname
             sbatch slurm.wolf.run
          fi

          cd $BASE_DIR  # not needed, but seems safer to do
      done

      # finalize the bundle script
      if [ $SETUP = 1 ]; then
         cd $BASE_DIR
         echo "wait" >> slurm.edison_bundle.amp${amp}_per${per}.run
      fi

   done
done

cd $BASE_DIR

