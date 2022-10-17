#!/bin/sh
for FILE in ./inputs/*
do
        echo "$FILE"
	./main "$FILE" &
done
