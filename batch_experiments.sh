#!/bin/bash

TESTSERIES=A
TESTNUM=1

while [[ -d lsgan/arch-1-test-${TESTNUM} ]]; do
    TESTNUM=$((TESTNUM+1))
done

MODEL="lsgan"
EXP_NAME="arch-\${ARCH}-test-${TESTSERIES}-${TESTNUM}"
ARCH_LIST=$(echo {1..7})
NUM_EPOCHS=500
EPOCHSIZE=25
EPOCHS_PER_CHECKPOINT=100
KEEP_CHECKPOINT_MODELS="--keep_all_checkpoints"
BATCH_SIZE=128
VERBOSITY="2"
EXTRA_ARGS="--force"


while ((1)); do
    echo "Running experiments on LSGAN with architectures 1 through 7...."
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
    
    for ARCH in ${ARCH_LIST}; do
	echo "Launching LSGAN model with Architecture #${ARCH}..."
	./run.py ${MODEL} ${EXP_NAME} -a ${ARCH} -e ${NUM_EPOCHS} -u ${EPOCHSIZE} -c ${EPOCHS_PER_CHECKPOINT} -b ${BATCH_SIZE} -v ${VERBPSITY} ${EXTRA_ARGS}
    done

    TESTNUM=$((TESTNUM+1))
    NUM_EPOCHS=$((NUM_EPOCHS+500))
    EPOCHSIZE=$((EPOCHSIZE+15))
    EPOCHS_PER_CHECKPOINT=$((EPOCHS_PER_CHECKPOINT+50))
done

echo "============================================================"
echo "        ALL EXPERIMENTS ARE COMPLETED    !!!!!!!!"
echo "============================================================"
