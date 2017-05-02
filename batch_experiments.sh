#!/bin/bash

TESTSERIES=A
TESTNUM=1

while [[ -d "lsgan/arch-1-test-${TESTNUM}*model*" ]]; do
    TESTNUM=$((TESTNUM+1))
done

ARCH_LIST=$(echo {1..7})
NUM_EPOCHS=250
EPOCHSIZE=25
EPOCHS_PER_CHECKPOINT=100
KEEP_CHECKPOINT_MODELS="--keep_all_checkpoints"
BATCH_SIZE=128
VERBOSITY=2
EXTRA_ARGS="--force"

iter=0
while (( iter++ >= 0 )); do

    ## Little hack so we can skip ahead without having to go through
    ## first 4 architectures after fixing the STOP file issue! :-)
    if [[ $iter == 1 ]]; then
	ARCH_LIST=$(echo {5..7})
    else
	ARCH_LIST=$(echo {1..7})
    fi


    MODEL="lsgan"
    EXP_NAME="arch-\${ARCH}-test-${TESTSERIES}-${TESTNUM}"
    
    echo "Running ITERATION $iter of experiments on LSGAN with architectures 1 through 7...."
    echo "Also doing one WGAN experiment per iteration :)"
    echo ""
    echo "Parameters:"
    echo " * MODEL                  = $MODEL"
    echo " * ARCH_LIST              = $ARCH_LIST"
    echo " * TESTSERIES             = $TESTSERIES"
    echo " * TESTNUM=               = $TESTNUM"
    echo " * NUM_EPOCHS             = $NUM_EPOCHS"
    echo " * EPOCHSIZE              = $EPOCHSIZE"
    echo " * EPOCHS_PER_CHECKPOINT  = $EPOCHS_PER_CHECKPOINT"
    echo " * KEEP_CHECKPOINT_MODELS = $KEEP_CHECKPOINT_MODELS"
    echo " * BATCH_SIZE             = $BATCH_SIZE"
    echo " * VERBOSITY              = $VERBOSITY"
    echo " * EXTRA_ARGS             = $EXTRA_ARGS"
    echo ""
    echo "Launching test ${TESTNUM} of series ${TESTSERIES}..."
    echo "Starting in 3, 2, 1...."
    sleep 3

    mkdir -p logs/lsgan
    mkdir -p logs/wgan
    mkdir -p logs/dcgan

    
    for ARCH in ${ARCH_LIST}; do

	THIS_NUM_EPOCHS=$NUM_EPOCHS
	THIS_EPOCHSIZE=$EPOCHSIZE
	
	if [[ $ARCH == 1 || $ARCH == 6 || $ARCH == 7 ]]; then
	    THIS_NUM_EPOCHS=$((2*NUM_EPOCHS))
	fi
    
	if [[ $ARCH == 4 ]]; then
	    THIS_NUM_EPOCHS=$((NUM_EPOCHS/4))
	    THIS_EPOCHSIZE=$((EPOCHSIZE/2))
	fi
	
	EXP_NAME="arch-${ARCH}-test-${TESTSERIES}-${TESTNUM}"

	echo "Launching LSGAN model with Architecture #${ARCH}..."
	echo "./run.py ${MODEL} ${EXP_NAME} -a ${ARCH} -e ${THIS_NUM_EPOCHS} -u ${THIS_EPOCHSIZE} -c ${EPOCHS_PER_CHECKPOINT} -b ${BATCH_SIZE} -v ${VERBOSITY} ${EXTRA_ARGS} | tee logs/${MODEL}/${EXP_NAME}"
	./run.py ${MODEL} ${EXP_NAME} -a ${ARCH} -e ${THIS_NUM_EPOCHS} -u ${THIS_EPOCHSIZE} -c ${EPOCHS_PER_CHECKPOINT} -b ${BATCH_SIZE} -v ${VERBOSITY} ${EXTRA_ARGS} | tee logs/${MODEL}/${EXP_NAME}
    done


    MODEL=wgan
    EXP_NAME="WGAN-test-${TESTSERIES}-${TESTNUM}"
    TOTAL_EPOCHS=$((100 + 200*iter))

    echo "Launching WGAN model..."
    echo "./run.py ${MODEL} ${EXP_NAME} -e ${TOTAL_EPOCHS} -b 128 -v 2 --force -c 100 | tee logs/${MODEL}/${EXP_NAME}"
    ./run.py ${MODEL} ${EXP_NAME} -e ${TOTAL_EPOCHS} -b 128 -v 2 --force -c 100 | tee logs/${MODEL}/${EXP_NAME}
								     

    TESTNUM=$((TESTNUM+1))
    NUM_EPOCHS=$((NUM_EPOCHS*2+250))
    EPOCHSIZE=$((EPOCHSIZE+25))
    EPOCHS_PER_CHECKPOINT=$((EPOCHS_PER_CHECKPOINT+100))
done

echo "============================================================"
echo "               ALL EXPERIMENTS ARE COMPLETED                "
echo "============================================================"

