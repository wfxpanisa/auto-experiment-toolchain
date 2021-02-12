#!/bin/bash

LD_PRELOAD=$1 $2 $3 & export KPID=$!

sleep $4

if [[ $5 -gt 0 ]]
then
	kill -30 $KPID
	sleep $5
fi

kill -9 $KPID
